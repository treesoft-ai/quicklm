#pragma once

#include <string>
#include <vector>

// A chat message: role ("system" / "user" / "assistant") + content.
struct ChatMessage {
    std::string role;
    std::string content;
};

// Renders a chat prompt from a model's own chat_template.jinja, using a minimal
// Jinja-subset interpreter.
//
// SCOPE: supports only the constructs the standard single/multi-turn chat path
// needs — {{ }} output with + / ~ string concat and |trim, {%- if/elif/else -%},
// {%- for message in messages -%} with loop.first / loop.index0, role comparisons,
// and the add_generation_prompt / enable_thinking flags. Whitespace-control markers
// ({%- / -%}) are honored. Anything outside this subset (macros, namespaces, the
// tools/vision/tool-response paths, reverse iteration, raise_exception) is NOT
// supported: render() reports failure and the caller falls back to a built-in
// default template that IS within the subset.
class ChatTemplate {
public:
    // Locate a template for the model: chat_template.jinja in model_dir, else the
    // "chat_template" field in tokenizer_config.json. `fallback` (an architecture's
    // default_chat_template()) is used when neither is present OR when the located
    // template uses unsupported constructs. Returns false only if no usable template
    // is available at all.
    bool load(const std::string& model_dir, const std::string& fallback);

    // Render `messages` to a prompt string. `add_generation_prompt` appends the
    // assistant generation primer; `enable_thinking` selects the thinking variant.
    // Returns true on success; on failure `out` is left empty.
    bool render(const std::vector<ChatMessage>& messages,
                bool add_generation_prompt, bool enable_thinking,
                std::string& out) const;

private:
    std::string template_src;       // the active (supported) template source
    bool have_template = false;

    // True if `src` uses only constructs render() supports.
    static bool is_supported(const std::string& src);
};
