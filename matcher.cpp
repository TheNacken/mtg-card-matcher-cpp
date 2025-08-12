#include "matcher.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <faiss/Index.h>
#include <faiss/IndexBinaryIVF.h>
#include <faiss/index_io.h>
#include <nlohmann/json.hpp>
#include <sqlite3.h>
#include <filesystem>
#include <stdexcept>
#include <map>
#include <vector>

namespace mtg {

static faiss::IndexBinaryIVF* index = nullptr;
static std::map<std::string, std::pair<int, int>> offsets;
static std::map<std::string, nlohmann::json> metadata;

void init(const std::string& indexPath, const std::string& sqlitePath) {
    if (!std::filesystem::exists(indexPath)) {
        throw std::runtime_error("Faiss index file does not exist: " + indexPath);
    }
    if (!std::filesystem::exists(sqlitePath)) {
        throw std::runtime_error("SQLite database file does not exist: " + sqlitePath);
    }

    index = dynamic_cast<faiss::IndexBinaryIVF*>(faiss::read_index_binary(indexPath.c_str()));
    if (!index) {
        throw std::runtime_error("Failed to load Faiss index from " + indexPath);
    }
    index->nprobe = 8;

    // Open SQLite and load offsets and metadata into memory
    sqlite3* db = nullptr;
    if (sqlite3_open(sqlitePath.c_str(), &db) != SQLITE_OK) {
        std::string err = sqlite3_errmsg(db);
        if (db) sqlite3_close(db);
        throw std::runtime_error("Failed to open SQLite DB: " + err);
    }

    // Load offsets
    {
        const char* sql = "SELECT card_id, start_idx, end_idx FROM offsets";
        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
            std::string err = sqlite3_errmsg(db);
            sqlite3_close(db);
            throw std::runtime_error("Failed to prepare offsets query: " + err);
        }
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            const unsigned char* c_id = sqlite3_column_text(stmt, 0);
            int start_idx = sqlite3_column_int(stmt, 1);
            int end_idx = sqlite3_column_int(stmt, 2);
            if (c_id) {
                offsets[reinterpret_cast<const char*>(c_id)] = {start_idx, end_idx};
            }
        }
        sqlite3_finalize(stmt);
    }

    // Load metadata
    {
        const char* sql = "SELECT card_id, meta_json FROM metadata";
        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
            std::string err = sqlite3_errmsg(db);
            sqlite3_close(db);
            throw std::runtime_error("Failed to prepare metadata query: " + err);
        }
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            const unsigned char* c_id = sqlite3_column_text(stmt, 0);
            const unsigned char* c_json = sqlite3_column_text(stmt, 1);
            if (c_id && c_json) {
                try {
                    nlohmann::json j = nlohmann::json::parse(reinterpret_cast<const char*>(c_json));
                    metadata[reinterpret_cast<const char*>(c_id)] = j;
                } catch (...) {
                    // Ignore malformed rows
                }
            }
        }
        sqlite3_finalize(stmt);
    }

    sqlite3_close(db);
}

std::string match(const std::string& imagePath) {
    auto t_start = std::chrono::high_resolution_clock::now();
    std::cout << "[DEBUG] Loading image: " << imagePath << std::endl;
    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        std::cout << "[DEBUG] Failed to load image." << std::endl;
        return "";
    }

    auto t_preprocess_start = std::chrono::high_resolution_clock::now();
    std::cout << "[DEBUG] Preprocessing image..." << std::endl;
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::resize(gray, gray, cv::Size(512, 512));
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    clahe->apply(gray, gray);
    auto t_preprocess_end = std::chrono::high_resolution_clock::now();
    std::cout << "[DEBUG] Preprocessing done. Time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t_preprocess_end - t_preprocess_start).count()
              << " ms" << std::endl;

    auto t_orb_start = std::chrono::high_resolution_clock::now();
    std::cout << "[DEBUG] Extracting ORB features..." << std::endl;
    cv::Ptr<cv::ORB> orb = cv::ORB::create(500);
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat query_descriptors;
    orb->detectAndCompute(gray, cv::noArray(), keypoints, query_descriptors);
    auto t_orb_end = std::chrono::high_resolution_clock::now();
    std::cout << "[DEBUG] ORB extraction done. Keypoints: " << keypoints.size()
              << ", Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(t_orb_end - t_orb_start).count()
              << " ms" << std::endl;
    if (query_descriptors.empty()) {
        std::cout << "[DEBUG] No descriptors found." << std::endl;
        return "";
    }

    query_descriptors = query_descriptors.clone(); // Ensure contiguous memory
    int n_queries = query_descriptors.rows;

    auto t_faiss_start = std::chrono::high_resolution_clock::now();
    std::cout << "[DEBUG] Performing Faiss search..." << std::endl;
    std::vector<int32_t> distances(n_queries * 2);
    std::vector<int64_t> indices(n_queries * 2);
    index->search(n_queries, query_descriptors.data, 2, distances.data(), indices.data());
    auto t_faiss_end = std::chrono::high_resolution_clock::now();
    std::cout << "[DEBUG] Faiss search done. Time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t_faiss_end - t_faiss_start).count()
              << " ms" << std::endl;

    auto t_match_start = std::chrono::high_resolution_clock::now();
    std::cout << "[DEBUG] Applying Lowe's ratio test..." << std::endl;
    std::vector<std::pair<int64_t, int32_t>> good_matches;
    const float ratio_thresh = 0.8f;
    for (int i = 0; i < n_queries; ++i) {
        int32_t d1 = distances[i * 2];
        int32_t d2 = distances[i * 2 + 1];
        if (d1 < ratio_thresh * d2) {
            good_matches.emplace_back(indices[i * 2], d1);
        }
    }
    std::cout << "[DEBUG] Good matches found: " << good_matches.size() << std::endl;
    if (good_matches.size() < 5) { // Minimum matches threshold
        std::cout << "[DEBUG] Not enough good matches." << std::endl;
        return "";
    }

    std::cout << "[DEBUG] Counting matches per card..." << std::endl;
    std::map<std::string, std::pair<int, float>> card_scores; // (count, avg_distance)
    for (const auto& [idx, dist] : good_matches) {
        for (const auto& [card_id, range] : offsets) {
            if (idx >= range.first && idx < range.second) {
                auto& score = card_scores[card_id];
                score.first += 1;
                score.second = (score.second * (score.first - 1) + static_cast<float>(dist)) / score.first; // Update avg distance
                break;
            }
        }
    }

    std::string best_card_id;
    float best_score = 0.0f;
    const int min_matches_per_card = 3;
    for (const auto& [card_id, score] : card_scores) {
        if (score.first < min_matches_per_card) continue;
        float s = score.first / (1.0f + score.second); // Score based on count and avg distance
        if (s > best_score) {
            best_score = s;
            best_card_id = card_id;
        }
    }
    auto t_match_end = std::chrono::high_resolution_clock::now();
    std::cout << "[DEBUG] Card matching done. Time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t_match_end - t_match_start).count()
              << " ms" << std::endl;

    auto t_end = std::chrono::high_resolution_clock::now();
    std::cout << "[DEBUG] Total match() time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count()
              << " ms" << std::endl;

    return best_card_id.empty() ? "" : best_card_id;
}

nlohmann::json get_metadata(const std::string& card_id) {
    auto it = metadata.find(card_id);
    if (it != metadata.end()) {
        return it->second;
    }
    return nlohmann::json{}; // Return empty JSON if not found
}

}