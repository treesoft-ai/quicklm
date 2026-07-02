#include "tokenizer.hpp"
#include "json_parser.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cctype>

// --- Base64 Helper ---

std::string base64_decode(const std::string& in) {
    std::string out;
    std::vector<int> T(256, -1);
    const char* b64_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    for (int i = 0; i < 64; i++) {
        T[static_cast<unsigned char>(b64_chars[i])] = i;
    }

    int val = 0, valb = -8;
    for (unsigned char c : in) {
        if (T[c] == -1) continue;
        val = (val << 6) + T[c];
        valb += 6;
        if (valb >= 0) {
            out.push_back(static_cast<char>((val >> valb) & 0xFF));
            valb -= 8;
        }
    }
    return out;
}

// --- GPT-2 byte<->unicode mapping ---
// Mirrors transformers' bytes_to_unicode(): a reversible map from bytes to
// printable unicode codepoints, each encoded here as a UTF-8 string.

static std::string codepoint_to_utf8(int cp) {
    std::string s;
    if (cp < 0x80) {
        s.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        s.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        s.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
    return s;
}

void Tokenizer::build_byte_unicode_maps() {
    // bs = printable ASCII + Latin-1 ranges; the rest get remapped to 256+n
    std::vector<int> bs;
    for (int b = 0x21; b <= 0x7E; ++b) bs.push_back(b);  // '!'..'~'
    for (int b = 0xA1; b <= 0xAC; ++b) bs.push_back(b);
    for (int b = 0xAE; b <= 0xFF; ++b) bs.push_back(b);

    std::vector<int> cs = bs;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n);
            ++n;
        }
    }

    for (size_t i = 0; i < bs.size(); ++i) {
        std::string u = codepoint_to_utf8(cs[i]);
        byte_to_unicode[bs[i]] = u;
        unicode_to_byte[u] = bs[i];
    }
}

// FNV-1a over a byte buffer, continuing from a running hash. Seed with
// FNV_OFFSET_BASIS for the first chunk. Cheap, dependency-free, and collision
// resistance is irrelevant here: the fingerprint guards against accidentally
// pairing two different tokenizers, not against an adversary.
static uint64_t fnv1a_accumulate(uint64_t hash, const std::string& bytes) {
    constexpr uint64_t FNV_PRIME = 1099511628211ULL;
    for (unsigned char c : bytes) {
        hash ^= c;
        hash *= FNV_PRIME;
    }
    return hash;
}

static constexpr uint64_t FNV_OFFSET_BASIS = 14695981039346656037ULL;

// --- Tokenizer Implementation ---

bool Tokenizer::load(const std::string& model_dir) {
    build_byte_unicode_maps();

    // Load vocab + added tokens from tokenizer.json
    std::string json_path = model_dir + "/tokenizer.json";
    std::ifstream json_file(json_path, std::ios::binary);
    if (!json_file.is_open()) {
        return false;
    }
    std::stringstream buffer;
    buffer << json_file.rdbuf();
    json_file.close();

    // Fingerprint everything this function reads off disk (see tokenizer.hpp).
    fingerprint_hash = fnv1a_accumulate(FNV_OFFSET_BASIS, buffer.str());

    try {
        JsonValue root = JsonParser::parse(buffer.str());
        const JsonValue& model_val = root["model"];
        const JsonValue& vocab_val = model_val["vocab"];

        if (vocab_val.is_object()) {
            for (const auto& pair : vocab_val.obj_val) {
                const std::string& token = pair.first;
                int id = static_cast<int>(pair.second.num_val);
                token_to_id[token] = id;
                id_to_token[id] = token;
            }
        }

        // Merges may live in tokenizer.json (model.merges) as an array of "A B" strings
        if (model_val.contains("merges")) {
            const JsonValue& merges_val = model_val["merges"];
            if (merges_val.is_array()) {
                int rank = 0;
                for (size_t i = 0; i < merges_val.arr_val.size(); ++i) {
                    bpe_ranks[merges_val[i].str_val] = rank++;
                }
            }
        }

        // Added/special tokens
        if (root.contains("added_tokens")) {
            const JsonValue& added_tokens = root["added_tokens"];
            if (added_tokens.is_array()) {
                for (size_t i = 0; i < added_tokens.arr_val.size(); ++i) {
                    const JsonValue& item = added_tokens[i];
                    std::string content = item["content"].str_val;
                    int id = static_cast<int>(item["id"].num_val);
                    special_tokens[content] = id;
                    // Make sure special tokens are decodable verbatim
                    id_to_token[id] = content;
                    if (token_to_id.find(content) == token_to_id.end()) {
                        token_to_id[content] = id;
                    }
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error parsing tokenizer.json: " << e.what() << std::endl;
        return false;
    }

    // Fallback: load merges.txt if not present in tokenizer.json
    if (bpe_ranks.empty()) {
        std::ifstream merges_file(model_dir + "/merges.txt", std::ios::binary);
        if (merges_file.is_open()) {
            // Slurp once so the fingerprint covers the exact bytes parsed below.
            std::stringstream merges_buf;
            merges_buf << merges_file.rdbuf();
            merges_file.close();
            fingerprint_hash = fnv1a_accumulate(fingerprint_hash, merges_buf.str());

            std::string line;
            int rank = 0;
            bool first = true;
            while (std::getline(merges_buf, line)) {
                if (!line.empty() && line.back() == '\r') line.pop_back();
                if (line.empty()) continue;
                if (first && line[0] == '#') { first = false; continue; } // version header
                first = false;
                bpe_ranks[line] = rank++;
            }
        }
    }

    // Resolve special token IDs
    auto get_special = [&](const std::string& name, int fallback) -> int {
        auto it = special_tokens.find(name);
        return (it != special_tokens.end()) ? it->second : fallback;
    };
    int endoftext = get_special("<|endoftext|>", -1);
    im_end_id = get_special("<|im_end|>", -1);
    // Chat models end turns with <|im_end|>; prefer it as EOS, fall back to <|endoftext|>.
    eos_id = (im_end_id != -1) ? im_end_id : endoftext;
    bos_id = -1;
    pad_id = (endoftext != -1) ? endoftext : eos_id;

    return !token_to_id.empty();
}

// Pre-tokenize raw UTF-8 text into GPT-2-style pieces, byte-level encoded.
// Approximates the GPT-2 regex: contractions, optional-leading-space words,
// optional-leading-space numbers, optional-leading-space symbol runs, and
// trailing whitespace runs. Non-ASCII (UTF-8 lead/continuation) bytes are
// treated as "letters" so multibyte characters stay grouped.
std::vector<std::string> Tokenizer::pre_tokenize(const std::string& text) const {
    std::vector<std::string> pieces;

    auto is_letter = [](unsigned char c) {
        return std::isalpha(c) || c >= 0x80;  // ASCII letters or any UTF-8 byte
    };
    auto is_digit = [](unsigned char c) { return std::isdigit(c) != 0; };
    auto is_space = [](unsigned char c) { return c == ' ' || c == '\t' || c == '\n' ||
                                                 c == '\r' || c == '\v' || c == '\f'; };

    auto byte_encode = [&](const std::string& raw) {
        std::string out;
        for (unsigned char b : raw) out += byte_to_unicode.at(b);
        return out;
    };

    size_t i = 0;
    size_t N = text.size();
    while (i < N) {
        // First, check for an exact special-token match at i.
        bool matched_special = false;
        for (const auto& pair : special_tokens) {
            const std::string& spec = pair.first;
            if (!spec.empty() && text.compare(i, spec.size(), spec) == 0) {
                pieces.push_back(spec);  // pushed verbatim (not byte-encoded)
                i += spec.size();
                matched_special = true;
                break;
            }
        }
        if (matched_special) continue;

        size_t start = i;
        unsigned char c = static_cast<unsigned char>(text[i]);

        // Optional single leading space attaches to the following word/number/symbol.
        bool lead_space = (c == ' ');
        size_t j = i + (lead_space ? 1 : 0);
        unsigned char d = (j < N) ? static_cast<unsigned char>(text[j]) : 0;

        if (lead_space && (j >= N || is_space(d))) {
            // A run of spaces with nothing (or more space) after: emit whitespace run.
            // Keep all but possibly-final space? GPT-2 keeps trailing spaces together.
            size_t k = i;
            while (k < N && is_space(static_cast<unsigned char>(text[k]))) ++k;
            pieces.push_back(byte_encode(text.substr(start, k - start)));
            i = k;
            continue;
        }

        if (is_letter(d)) {
            i = j;
            while (i < N && is_letter(static_cast<unsigned char>(text[i]))) ++i;
            pieces.push_back(byte_encode(text.substr(start, i - start)));
        } else if (is_digit(d)) {
            i = j;
            while (i < N && is_digit(static_cast<unsigned char>(text[i]))) ++i;
            pieces.push_back(byte_encode(text.substr(start, i - start)));
        } else if (is_space(c)) {
            // Whitespace run (no following word).
            size_t k = i;
            while (k < N && is_space(static_cast<unsigned char>(text[k]))) ++k;
            pieces.push_back(byte_encode(text.substr(start, k - start)));
            i = k;
        } else {
            // Symbol/punctuation run (with optional leading space already consumed).
            i = j;
            while (i < N) {
                unsigned char e = static_cast<unsigned char>(text[i]);
                if (is_letter(e) || is_digit(e) || is_space(e)) break;
                ++i;
            }
            if (i == start) ++i;  // safety: always advance
            pieces.push_back(byte_encode(text.substr(start, i - start)));
        }
    }
    return pieces;
}

// Split a byte-level-encoded piece into a vector of single "characters", where
// each character is one entry of the byte_to_unicode map (1..3 UTF-8 bytes).
static std::vector<std::string> split_unicode_chars(const std::string& s) {
    std::vector<std::string> chars;
    size_t i = 0;
    while (i < s.size()) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        size_t len = 1;
        if ((c & 0x80) == 0) len = 1;
        else if ((c & 0xE0) == 0xC0) len = 2;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xF8) == 0xF0) len = 4;
        chars.push_back(s.substr(i, len));
        i += len;
    }
    return chars;
}

std::vector<int> Tokenizer::bpe_encode(const std::string& piece) const {
    // Special tokens are passed through directly.
    auto spec_it = special_tokens.find(piece);
    if (spec_it != special_tokens.end()) {
        return {spec_it->second};
    }

    std::vector<std::string> parts = split_unicode_chars(piece);
    if (parts.empty()) return {};

    // Iteratively merge the highest-priority (lowest-rank) adjacent pair.
    while (parts.size() > 1) {
        int best_rank = -1;
        size_t best_idx = 0;
        for (size_t j = 0; j + 1 < parts.size(); ++j) {
            std::string pair = parts[j] + " " + parts[j + 1];
            auto it = bpe_ranks.find(pair);
            if (it != bpe_ranks.end()) {
                if (best_rank == -1 || it->second < best_rank) {
                    best_rank = it->second;
                    best_idx = j;
                }
            }
        }
        if (best_rank == -1) break;  // no more merges
        parts[best_idx] = parts[best_idx] + parts[best_idx + 1];
        parts.erase(parts.begin() + best_idx + 1);
    }

    std::vector<int> ids;
    ids.reserve(parts.size());
    for (const auto& p : parts) {
        auto it = token_to_id.find(p);
        if (it != token_to_id.end()) {
            ids.push_back(it->second);
        } else {
            // Fall back to per-character lookup (should be rare; base vocab has all bytes).
            for (const auto& ch : split_unicode_chars(p)) {
                auto cit = token_to_id.find(ch);
                if (cit != token_to_id.end()) ids.push_back(cit->second);
            }
        }
    }
    return ids;
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    std::vector<std::string> pieces = pre_tokenize(text);
    std::vector<int> ids;
    for (const auto& piece : pieces) {
        std::vector<int> piece_ids = bpe_encode(piece);
        ids.insert(ids.end(), piece_ids.begin(), piece_ids.end());
    }
    return ids;
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
    // Concatenate the byte-level token strings, then map each unicode "char"
    // back to its original raw byte to reconstruct UTF-8 text.
    std::string encoded;
    for (int id : ids) {
        auto it = id_to_token.find(id);
        if (it != id_to_token.end()) {
            encoded += it->second;
        }
    }

    std::string out;
    for (const auto& ch : split_unicode_chars(encoded)) {
        auto bit = unicode_to_byte.find(ch);
        if (bit != unicode_to_byte.end()) {
            out.push_back(static_cast<char>(bit->second));
        } else {
            // Special-token text (e.g. "<|im_end|>") isn't byte-encoded; emit as-is.
            out += ch;
        }
    }
    return out;
}
