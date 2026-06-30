#pragma once

#include <string>
#include <vector>
#include <string_view>
#include <stdexcept>
#include <cctype>
#include <cstdlib>

// A zero-dependency, lightweight JSON parser in C++17.
// Parsed objects represent the JSON structure and can be queried.
struct JsonValue {
    enum Type { Null, Bool, Number, String, Array, Object };
    Type type = Null;

    bool bool_val = false;
    double num_val = 0.0;
    std::string str_val;
    std::vector<JsonValue> arr_val;
    std::vector<std::pair<std::string, JsonValue>> obj_val;

    bool is_null() const { return type == Null; }
    bool is_bool() const { return type == Bool; }
    bool is_number() const { return type == Number; }
    bool is_string() const { return type == String; }
    bool is_array() const { return type == Array; }
    bool is_object() const { return type == Object; }

    const JsonValue& operator[](const std::string& key) const {
        if (type == Object) {
            for (const auto& pair : obj_val) {
                if (pair.first == key) {
                    return pair.second;
                }
            }
        }
        static JsonValue null_val;
        return null_val;
    }

    const JsonValue& operator[](size_t index) const {
        if (type == Array && index < arr_val.size()) {
            return arr_val[index];
        }
        static JsonValue null_val;
        return null_val;
    }

    bool contains(const std::string& key) const {
        if (type == Object) {
            for (const auto& pair : obj_val) {
                if (pair.first == key) {
                    return true;
                }
            }
        }
        return false;
    }
};

class JsonParser {
public:
    static JsonValue parse(std::string_view json_str) {
        size_t idx = 0;
        skip_ws(json_str, idx);
        JsonValue val = parse_value(json_str, idx);
        skip_ws(json_str, idx);
        if (idx < json_str.size()) {
            throw std::runtime_error("Unexpected characters after JSON content");
        }
        return val;
    }

private:
    static void skip_ws(std::string_view str, size_t& idx) {
        while (idx < str.size() && std::isspace(static_cast<unsigned char>(str[idx]))) {
            ++idx;
        }
    }

    static JsonValue parse_value(std::string_view str, size_t& idx) {
        skip_ws(str, idx);
        if (idx >= str.size()) {
            throw std::runtime_error("Unexpected end of JSON string");
        }

        char c = str[idx];
        if (c == '{') {
            return parse_object(str, idx);
        } else if (c == '[') {
            return parse_array(str, idx);
        } else if (c == '"') {
            return parse_string(str, idx);
        } else if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) {
            return parse_number(str, idx);
        } else if (str.compare(idx, 4, "true") == 0) {
            idx += 4;
            JsonValue val;
            val.type = JsonValue::Bool;
            val.bool_val = true;
            return val;
        } else if (str.compare(idx, 5, "false") == 0) {
            idx += 5;
            JsonValue val;
            val.type = JsonValue::Bool;
            val.bool_val = false;
            return val;
        } else if (str.compare(idx, 4, "null") == 0) {
            idx += 4;
            JsonValue val;
            val.type = JsonValue::Null;
            return val;
        }

        throw std::runtime_error("Invalid JSON character at index " + std::to_string(idx) + ": '" + c + "'");
    }

    static JsonValue parse_object(std::string_view str, size_t& idx) {
        idx++; // consume '{'
        JsonValue val;
        val.type = JsonValue::Object;

        while (true) {
            skip_ws(str, idx);
            if (idx >= str.size()) {
                throw std::runtime_error("Unterminated object");
            }
            if (str[idx] == '}') {
                idx++;
                break;
            }

            if (str[idx] != '"') {
                throw std::runtime_error("Expected string key in object at index " + std::to_string(idx));
            }

            JsonValue key_val = parse_string(str, idx);
            std::string key = key_val.str_val;

            skip_ws(str, idx);
            if (idx >= str.size() || str[idx] != ':') {
                throw std::runtime_error("Expected ':' after key in object");
            }
            idx++; // consume ':'

            JsonValue value = parse_value(str, idx);
            val.obj_val.emplace_back(std::move(key), std::move(value));

            skip_ws(str, idx);
            if (idx >= str.size()) {
                throw std::runtime_error("Unterminated object");
            }

            if (str[idx] == ',') {
                idx++;
            } else if (str[idx] == '}') {
                idx++;
                break;
            } else {
                throw std::runtime_error("Expected ',' or '}' in object at index " + std::to_string(idx));
            }
        }

        return val;
    }

    static JsonValue parse_array(std::string_view str, size_t& idx) {
        idx++; // consume '['
        JsonValue val;
        val.type = JsonValue::Array;

        while (true) {
            skip_ws(str, idx);
            if (idx >= str.size()) {
                throw std::runtime_error("Unterminated array");
            }
            if (str[idx] == ']') {
                idx++;
                break;
            }

            val.arr_val.push_back(parse_value(str, idx));

            skip_ws(str, idx);
            if (idx >= str.size()) {
                throw std::runtime_error("Unterminated array");
            }

            if (str[idx] == ',') {
                idx++;
            } else if (str[idx] == ']') {
                idx++;
                break;
            } else {
                throw std::runtime_error("Expected ',' or ']' in array at index " + std::to_string(idx));
            }
        }

        return val;
    }

    static JsonValue parse_string(std::string_view str, size_t& idx) {
        idx++; // consume '"'
        std::string res;
        while (idx < str.size()) {
            char c = str[idx];
            if (c == '"') {
                idx++;
                JsonValue val;
                val.type = JsonValue::String;
                val.str_val = std::move(res);
                return val;
            } else if (c == '\\') {
                idx++;
                if (idx >= str.size()) {
                    throw std::runtime_error("Unterminated string escape sequence");
                }
                char esc = str[idx];
                if (esc == '"') res += '"';
                else if (esc == '\\') res += '\\';
                else if (esc == '/') res += '/';
                else if (esc == 'b') res += '\b';
                else if (esc == 'f') res += '\f';
                else if (esc == 'n') res += '\n';
                else if (esc == 'r') res += '\r';
                else if (esc == 't') res += '\t';
                else if (esc == 'u') {
                    // Skip unicode hex sequence for simplicity (or decode it if needed)
                    // Qwen vocabulary strings are often stored as standard bytes or basic escapes
                    if (idx + 4 >= str.size()) {
                        throw std::runtime_error("Incomplete unicode escape");
                    }
                    res += "\\u";
                    res += std::string(str.substr(idx + 1, 4));
                    idx += 4;
                } else {
                    res += esc;
                }
            } else {
                res += c;
            }
            idx++;
        }
        throw std::runtime_error("Unterminated string");
    }

    static JsonValue parse_number(std::string_view str, size_t& idx) {
        size_t start = idx;
        if (str[idx] == '-') {
            idx++;
        }
        while (idx < str.size() && (std::isdigit(static_cast<unsigned char>(str[idx])) || 
                                    str[idx] == '.' || str[idx] == 'e' || str[idx] == 'E' || 
                                    str[idx] == '+' || str[idx] == '-')) {
            idx++;
        }
        std::string num_str(str.substr(start, idx - start));
        char* end;
        double val = std::strtod(num_str.c_str(), &end);
        if (end == num_str.c_str()) {
            throw std::runtime_error("Invalid number format: " + num_str);
        }
        JsonValue jval;
        jval.type = JsonValue::Number;
        jval.num_val = val;
        return jval;
    }
};
