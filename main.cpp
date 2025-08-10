#include "matcher.h"
#include <iostream>
#include <nlohmann/json.hpp>

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: ./matcher_test indexPath sqlitePath imagePath" << std::endl;
        return 1;
    }

    std::string indexPath = argv[1];
    std::string sqlitePath = argv[2];
    std::string imagePath = argv[3];

    try {
        mtg::init(indexPath, sqlitePath);
        std::string card_id = mtg::match(imagePath);

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
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}