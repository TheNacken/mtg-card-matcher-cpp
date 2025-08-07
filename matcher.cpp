#include "matcher.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <H5Cpp.h>
#include <faiss/IndexBinaryIVF.h>
#include <faiss/index_io.h>
#include <nlohmann/json.hpp>
#include <filesystem>
#include <stdexcept>
#include <map>
#include <vector>

namespace mtg {

static faiss::IndexBinaryIVF* index = nullptr;
static H5::H5File* hdf5_file = nullptr;
static std::vector<uint8_t> descriptors;
static std::map<std::string, std::pair<int, int>> offsets;
static std::map<std::string, nlohmann::json> metadata;

void init(const std::string& indexPath, const std::string& hdf5Path) {
    if (!std::filesystem::exists(indexPath)) {
        throw std::runtime_error("Faiss index file does not exist: " + indexPath);
    }
    if (!std::filesystem::exists(hdf5Path)) {
        throw std::runtime_error("HDF5 file does not exist: " + hdf5Path);
    }

    index = dynamic_cast<faiss::IndexBinaryIVF*>(faiss::read_index_binary(indexPath.c_str()));
    if (!index) {
        throw std::runtime_error("Failed to load Faiss index from " + indexPath);
    }
    index->nprobe = 8;

    hdf5_file = new H5::H5File(hdf5Path, H5F_ACC_RDONLY);

    std::cout << "Reading descriptors dataset..." << std::endl;
    H5::DataSet desc_dataset = hdf5_file->openDataSet("descriptors");
    H5::DataSpace desc_dataspace = desc_dataset.getSpace();
    hsize_t dims[2];
    desc_dataspace.getSimpleExtentDims(dims, nullptr);
    std::cout << "Descriptors shape: (" << dims[0] << ", " << dims[1] << ")" << std::endl;
    descriptors.resize(dims[0] * dims[1]);
    desc_dataset.read(descriptors.data(), H5::PredType::NATIVE_UINT8);

    std::cout << "Reading offsets dataset..." << std::endl;
    H5::DataSet offsets_dataset = hdf5_file->openDataSet("offsets");
    H5::StrType str_type(H5::PredType::C_S1, H5T_VARIABLE);
    std::string offsets_json;
    offsets_dataset.read(offsets_json, str_type);
    nlohmann::json offsets_data = nlohmann::json::parse(offsets_json);
    for (auto& [card_id, range] : offsets_data.items()) {
        offsets[card_id] = {range[0].get<int>(), range[1].get<int>()};
    }

    std::cout << "Reading metadata dataset..." << std::endl;
    H5::DataSet metadata_dataset = hdf5_file->openDataSet("metadata");
    std::string metadata_json;
    metadata_dataset.read(metadata_json, str_type);
    nlohmann::json metadata_data = nlohmann::json::parse(metadata_json);
    for (auto& [card_id, meta] : metadata_data.items()) {
        metadata[card_id] = meta;
    }
}

std::string match(const std::string& imagePath) {
    // Load image
    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        return "";
    }

    // Preprocess image
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::resize(gray, gray, cv::Size(512, 512));
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    clahe->apply(gray, gray);

    // Extract ORB features
    cv::Ptr<cv::ORB> orb = cv::ORB::create(500);
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat query_descriptors;
    orb->detectAndCompute(gray, cv::noArray(), keypoints, query_descriptors);
    if (query_descriptors.empty()) {
        return "";
    }

    // Ensure descriptors are contiguous and in the right format
    query_descriptors = query_descriptors.clone(); // Ensure contiguous memory
    int n_queries = query_descriptors.rows;

    // Perform Faiss search (k=2 for Lowe's ratio test)
    std::vector<int> distances(n_queries * 2);
    std::vector<faiss::idx_t> indices(n_queries * 2);
    index->search(n_queries, query_descriptors.data, 2, distances.data(), indices.data());

    // Apply Lowe's ratio test
    std::vector<std::pair<faiss::idx_t, int>> good_matches;
    const float ratio_thresh = 0.8f;
    for (int i = 0; i < n_queries; ++i) {
        int d1 = distances[i * 2];
        int d2 = distances[i * 2 + 1];
        if (d1 < ratio_thresh * d2) {
            good_matches.emplace_back(indices[i * 2], d1);
        }
    }

    if (good_matches.size() < 5) { // Minimum matches threshold
        return "";
    }

    // Count matches per card
    std::map<std::string, std::pair<int, float>> card_scores; // (count, avg_distance)
    for (const auto& [idx, dist] : good_matches) {
        for (const auto& [card_id, range] : offsets) {
            if (idx >= range.first && idx < range.second) {
                auto& score = card_scores[card_id];
                score.first += 1;
                score.second = (score.second * (score.first - 1) + dist) / score.first; // Update avg distance
                break;
            }
        }
    }

    // Find best card
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