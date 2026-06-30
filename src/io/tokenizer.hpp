#pragma once

#include <string>
#include <vector>
#include <unordered_map>

class Tokenizer {
public:
    Tokenizer() = default;
    ~Tokenizer() = default;

    // Load from tokenizer.json (+ merges.txt) inside model_dir
    bool load(const std::string& model_dir);

    // Encode a prompt string into a sequence of token IDs
    std::vector<int> encode(const std::string& text) const;

    // Decode a sequence of token IDs back into a string (UTF-8 bytes)
    std::string decode(const std::vector<int>& ids) const;

    // Retrieve vocabulary size
    int vocab_size() const { return static_cast<int>(id_to_token.size()); }

    // Special token IDs
    int pad_token_id() const { return pad_id; }
    int bos_token_id() const { return bos_id; }
    int eos_token_id() const { return eos_id; }
    int im_end_token_id() const { return im_end_id; }

private:
    // Core vocabulary mapping: byte-level token string -> ID
    std::unordered_map<std::string, int> token_to_id;
    // Reverse mapping: ID -> byte-level token string
    std::unordered_map<int, std::string> id_to_token;

    // Special tokens mapping (content -> ID)
    std::unordered_map<std::string, int> special_tokens;

    // BPE merge ranks: "A B" merged pair -> rank (lower = higher priority)
    std::unordered_map<std::string, int> bpe_ranks;

    // GPT-2 byte-level mapping: raw byte -> unicode "char" string, and reverse
    std::unordered_map<int, std::string> byte_to_unicode;     // 0..255 -> utf8 string of 1 codepoint
    std::unordered_map<std::string, int> unicode_to_byte;     // utf8 codepoint string -> byte

    int pad_id = -1;
    int bos_id = -1;
    int eos_id = -1;
    int im_end_id = -1;

    void build_byte_unicode_maps();

    // Pre-tokenize raw UTF-8 text into GPT-2-style pieces (already byte-level encoded)
    std::vector<std::string> pre_tokenize(const std::string& text) const;
    // Run BPE on a single byte-level-encoded piece, returning token IDs
    std::vector<int> bpe_encode(const std::string& piece) const;
};

// Base64 helper defined in tokenizer.cpp
std::string base64_decode(const std::string& in);
