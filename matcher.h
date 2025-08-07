#ifndef MATCHER_H
#define MATCHER_H

#include <string>
#include <nlohmann/json.hpp>

namespace mtg {

void init(const std::string& indexPath, const std::string& hdf5Path);
std::string match(const std::string& imagePath);
nlohmann::json get_metadata(const std::string& card_id);

} // namespace mtg

#endif // MATCHER_H