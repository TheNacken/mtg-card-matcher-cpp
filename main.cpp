#include "matcher.h"
#include <iostream>
#include <nlohmann/json.hpp>

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: ./matcher_test indexPath hdf5Path imagePath" << std::endl;
        return 1;
    }

    std::string indexPath = argv[1];
    std::string hdf5Path = argv[2];
    std::string imagePath = argv[3];

    try {
        mtg::init(indexPath, hdf5Path);
        std::string card_id = mtg::match(imagePath);

        if (!card_id.empty()) {
            nlohmann::json meta = mtg::get_metadata(card_id);  // Access metadata via function
            if (!meta.is_null()) {
                std::string card_name = meta["card_name"].get<std::string>();
                std::string set_name = meta["set_name"].get<std::string>();
                std::string set_code = meta["set_code"].get<std::string>();

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
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}