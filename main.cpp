#include "matcher.h"
#include <iostream>
#include <nlohmann/json.hpp>
#include <fstream>
#include <iterator>

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: ./matcher_test indexPath sqlitePath imagePath" << std::endl;
        return 1;
    }

    std::string indexPath = argv[1];
    std::string sqlitePath = argv[2];
    std::string imagePath = argv[3];

    try {
        auto t_start = std::chrono::high_resolution_clock::now();
        auto t_init_start = t_start;
        mtg::init(indexPath, sqlitePath);
        auto t_init_end = std::chrono::high_resolution_clock::now();
        auto init_time = std::chrono::duration_cast<std::chrono::milliseconds>(t_init_end - t_init_start).count();

        // Load image file into memory buffer
        std::ifstream file(imagePath, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Failed to open image file: " + imagePath);
        }
        std::vector<unsigned char> imageData((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

        auto t_match_start = t_init_end;
        std::string card_id = mtg::match(imageData);
        auto t_match_end = std::chrono::high_resolution_clock::now();
        auto match_time = std::chrono::duration_cast<std::chrono::milliseconds>(t_match_end - t_match_start).count();

        auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(t_match_end - t_start).count();

        if (!card_id.empty()) {
            nlohmann::json meta = mtg::get_metadata(card_id);  // Access metadata via function
            if (!meta.is_null()) {
                std::string card_name = meta.value("card_name", std::string("Unknown"));
                std::string set_name = meta.value("set_name", std::string("Unknown"));
                std::string set_code = meta.value("set_code", std::string("?"));

                // Dummy confidence (replace with actual logic if available)
                float confidence = 0.95f;

                std::cout << "Matched Card: " << card_name << "\n"
                          << "Set: " << set_name << " (" << set_code << ")\n"
                          << "Confidence: " << confidence << std::endl;
            } else {
                std::cout << "Metadata not found for card ID: " << card_id << std::endl;
            }
        } else {
            std::cout << "No match found" << std::endl;
        }
        std::cout << "Initialization time: " << init_time << " ms" << std::endl;
        std::cout << "Matching time: " << match_time << " ms" << std::endl;
        std::cout << "Total processing time: " << total_time << " ms" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}