#ifndef MATCHER_H
#define MATCHER_H

#include <string>
#include <nlohmann/json.hpp>
#include <vector>

namespace mtg {

void init(const std::string& indexPath, const std::string& sqlitePath);
std::string match(const std::vector<unsigned char>& imageData);
nlohmann::json get_metadata(const std::string& card_id);

} // namespace mtg

#endif // MATCHER_H