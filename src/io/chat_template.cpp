#include "chat_template.hpp"
#include "json_parser.hpp"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cstdio>
#include <cstdlib>

namespace fs = std::filesystem;

// ============================================================================
// Minimal Jinja-subset renderer
// ----------------------------------------------------------------------------
// The grammar handled here is intentionally tiny — exactly what the standard chat
// path needs. The template is tokenized into a flat list of nodes; control flow
// (if / for) is matched into spans and walked recursively. Expressions are limited
// to string literals, a handful of variables/attributes, string concatenation
// (+ / ~), and the |trim filter. Booleans appear only in if-conditions.
// ============================================================================

namespace {

// ---- Value model -----------------------------------------------------------
// During rendering, an expression evaluates to either a string or a bool.
struct Value {
    bool is_bool = false;
    bool b = false;
    std::string s;
    static Value Str(std::string v) { Value x; x.s = std::move(v); return x; }
    static Value Bool(bool v) { Value x; x.is_bool = true; x.b = v; return x; }
};

struct RenderError : std::runtime_error {
    explicit RenderError(const std::string& m) : std::runtime_error(m) {}
};

// ---- Rendering context -----------------------------------------------------
struct Env {
    const std::vector<ChatMessage>* messages = nullptr;
    bool add_generation_prompt = false;
    bool enable_thinking = false;

    // Active loop state (single-level for loop over messages).
    const ChatMessage* cur_msg = nullptr;
    int loop_index0 = 0;
    bool loop_first = false;
};

std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

// ---- Tokenizer (template source -> nodes) ----------------------------------
struct Node {
    enum Kind { Text, Output, Stmt } kind;
    std::string body;           // Text: literal; Output/Stmt: trimmed inner expr
    bool trim_left = false;     // {%-  /  {{-
    bool trim_right = false;    // -%}  /  -}}
};

std::vector<Node> tokenize(const std::string& src) {
    std::vector<Node> nodes;
    size_t i = 0, n = src.size();
    std::string text;
    auto flush_text = [&]() {
        if (!text.empty()) { nodes.push_back({Node::Text, text, false, false}); text.clear(); }
    };
    while (i < n) {
        if (i + 1 < n && src[i] == '{' && (src[i + 1] == '{' || src[i + 1] == '%')) {
            char kind = src[i + 1];
            // The char preceding the final '}' differs by tag kind:
            //   output    {{ ... }}  -> '}'
            //   statement {% ... %}  -> '%'
            char close = (kind == '{') ? '}' : '%';
            flush_text();
            size_t j = i + 2;
            bool tl = false;
            if (j < n && src[j] == '-') { tl = true; ++j; }
            // find closing  -%}/%}  or  -}}/}}
            std::string inner;
            bool tr = false;
            while (j < n) {
                if (src[j] == '-' && j + 2 < n && src[j + 1] == close && src[j + 2] == '}') {
                    tr = true; j += 3; break;
                }
                if (src[j] == close && j + 1 < n && src[j + 1] == '}') {
                    j += 2; break;
                }
                inner += src[j++];
            }
            Node nd;
            nd.kind = (kind == '{') ? Node::Output : Node::Stmt;
            nd.body = trim(inner);
            nd.trim_left = tl;
            nd.trim_right = tr;
            nodes.push_back(nd);
            i = j;
        } else {
            text += src[i++];
        }
    }
    flush_text();
    return nodes;
}

// ---- Expression evaluation -------------------------------------------------
std::string unquote(const std::string& tok) {
    // Single- or double-quoted literal; unescape \n \t \\ \' \"
    char q = tok[0];
    std::string out;
    for (size_t i = 1; i + 1 < tok.size(); ++i) {
        char c = tok[i];
        if (c == '\\' && i + 1 < tok.size()) {
            char e = tok[++i];
            if (e == 'n') out += '\n';
            else if (e == 't') out += '\t';
            else if (e == 'r') out += '\r';
            else if (e == '\\') out += '\\';
            else if (e == q) out += q;
            else { out += '\\'; out += e; }
        } else {
            out += c;
        }
    }
    return out;
}

bool is_quote(char c) { return c == '\'' || c == '"'; }

// Split an expression on a top-level operator char (not inside quotes). Returns
// the pieces; `ops` receives the operator that preceded each piece after the first.
std::vector<std::string> split_top(const std::string& expr, const std::string& op_chars,
                                   std::vector<char>& ops) {
    std::vector<std::string> parts;
    std::string cur;
    char inq = 0;
    for (size_t i = 0; i < expr.size(); ++i) {
        char c = expr[i];
        if (inq) {
            cur += c;
            if (c == inq && expr[i - 1] != '\\') inq = 0;
            continue;
        }
        if (is_quote(c)) { inq = c; cur += c; continue; }
        if (op_chars.find(c) != std::string::npos) {
            parts.push_back(cur); cur.clear(); ops.push_back(c);
        } else {
            cur += c;
        }
    }
    parts.push_back(cur);
    return parts;
}

// Evaluate a single primary term (no + / ~) to a Value.
Value eval_primary(const std::string& raw, const Env& env);

// Evaluate a string-valued expression with + / ~ concatenation and |trim.
std::string eval_string(const std::string& expr_in, const Env& env) {
    std::string expr = trim(expr_in);

    // Trailing |trim filter (only filter supported).
    bool do_trim = false;
    {
        std::vector<char> ops;
        std::vector<std::string> pipe = split_top(expr, "|", ops);
        if (pipe.size() >= 2) {
            for (size_t k = 1; k < pipe.size(); ++k) {
                if (trim(pipe[k]) != "trim") throw RenderError("unsupported filter: " + pipe[k]);
            }
            do_trim = true;
            expr = trim(pipe[0]);
        }
    }

    std::vector<char> ops;
    std::vector<std::string> parts = split_top(expr, "+~", ops);
    std::string out;
    for (auto& p : parts) {
        Value v = eval_primary(trim(p), env);
        if (v.is_bool) throw RenderError("boolean used in string context");
        out += v.s;
    }
    return do_trim ? trim(out) : out;
}

Value eval_primary(const std::string& raw, const Env& env) {
    std::string t = trim(raw);
    if (t.empty()) return Value::Str("");
    if (is_quote(t[0])) return Value::Str(unquote(t));

    if (t == "add_generation_prompt") return Value::Bool(env.add_generation_prompt);
    if (t == "enable_thinking") return Value::Bool(env.enable_thinking);
    if (t == "loop.first") return Value::Bool(env.loop_first);
    if (t == "loop.index0") return Value::Str(std::to_string(env.loop_index0));
    if (t == "true") return Value::Bool(true);
    if (t == "false") return Value::Bool(false);

    if (t == "message.role" || t == "message['role']") {
        if (!env.cur_msg) throw RenderError("message.role outside loop");
        return Value::Str(env.cur_msg->role);
    }
    if (t == "message.content" || t == "message['content']") {
        if (!env.cur_msg) throw RenderError("message.content outside loop");
        return Value::Str(env.cur_msg->content);
    }

    throw RenderError("unsupported expression: " + t);
}

// Evaluate an if-condition to bool. Supports: a single primary, ==/!= comparison
// of two primaries, and a bare variable/flag (truthiness).
bool eval_condition(const std::string& cond_in, const Env& env) {
    std::string cond = trim(cond_in);

    // Equality / inequality.
    for (const char* op : {"==", "!="}) {
        size_t pos = std::string::npos;
        char inq = 0;
        for (size_t i = 0; i + 1 < cond.size(); ++i) {
            char c = cond[i];
            if (inq) { if (c == inq && cond[i - 1] != '\\') inq = 0; continue; }
            if (is_quote(c)) { inq = c; continue; }
            if (cond[i] == op[0] && cond[i + 1] == op[1]) { pos = i; break; }
        }
        if (pos != std::string::npos) {
            Value l = eval_primary(cond.substr(0, pos), env);
            Value r = eval_primary(cond.substr(pos + 2), env);
            std::string ls = l.is_bool ? (l.b ? "true" : "false") : l.s;
            std::string rs = r.is_bool ? (r.b ? "true" : "false") : r.s;
            bool eq = (ls == rs);
            return (op[0] == '=') ? eq : !eq;
        }
    }

    Value v = eval_primary(cond, env);
    if (v.is_bool) return v.b;
    return !v.s.empty();
}

// ---- Statement parsing helpers ---------------------------------------------
std::string stmt_keyword(const std::string& body) {
    size_t sp = body.find_first_of(" \t");
    return body.substr(0, sp);
}

// Find the matching {% endX %} for the block opened at nodes[open], honoring nesting.
// `enders` lists statement keywords that close/continue the block at the SAME level.
size_t find_block_end(const std::vector<Node>& nodes, size_t open,
                      const std::string& open_kw, const std::vector<std::string>& enders) {
    int depth = 0;
    for (size_t i = open + 1; i < nodes.size(); ++i) {
        if (nodes[i].kind != Node::Stmt) continue;
        std::string kw = stmt_keyword(nodes[i].body);
        if (kw == open_kw) { depth++; continue; }
        if (kw == "end" + open_kw) {
            if (depth == 0) return i;
            depth--;
            continue;
        }
        if (depth == 0) {
            for (const auto& e : enders) if (kw == e) return i;
        }
    }
    throw RenderError("unterminated {% " + open_kw + " %}");
}

void render_range(const std::vector<Node>& nodes, size_t lo, size_t hi,
                  Env& env, std::string& out);

// Render an if/elif/else chain starting at nodes[start] (an "if"). Returns the
// index just past the matching endif.
size_t render_if(const std::vector<Node>& nodes, size_t start, Env& env, std::string& out) {
    size_t cur = start;
    bool taken = false;
    while (true) {
        const Node& head = nodes[cur];
        std::string kw = stmt_keyword(head.body);
        size_t next = find_block_end(nodes, cur, "if",
                                     {"elif", "else"});
        // Decide whether this branch runs.
        bool run = false;
        if (!taken) {
            if (kw == "if" || kw == "elif") {
                std::string cond = trim(head.body.substr(kw.size()));
                run = eval_condition(cond, env);
            } else { // else
                run = true;
            }
        }
        if (run) { render_range(nodes, cur + 1, next, env, out); taken = true; }

        std::string nk = stmt_keyword(nodes[next].body);
        if (nk == "endif") return next + 1;
        cur = next; // advance to elif/else and continue
    }
}

// Render a {% for message in messages %} loop. Returns index past endfor.
size_t render_for(const std::vector<Node>& nodes, size_t start, Env& env, std::string& out) {
    const Node& head = nodes[start];
    // Only "for message in messages" is supported.
    std::string body = trim(head.body.substr(3)); // after "for"
    if (body != "message in messages")
        throw RenderError("unsupported for-loop: " + body);
    size_t endfor = find_block_end(nodes, start, "for", {});

    const auto& msgs = *env.messages;
    for (size_t mi = 0; mi < msgs.size(); ++mi) {
        env.cur_msg = &msgs[mi];
        env.loop_index0 = static_cast<int>(mi);
        env.loop_first = (mi == 0);
        render_range(nodes, start + 1, endfor, env, out);
    }
    env.cur_msg = nullptr;
    return endfor + 1;
}

// Render nodes[lo, hi), applying whitespace-control trimming between tags.
void render_range(const std::vector<Node>& nodes, size_t lo, size_t hi,
                  Env& env, std::string& out) {
    size_t i = lo;
    while (i < hi) {
        const Node& nd = nodes[i];
        if (nd.kind == Node::Text) {
            std::string t = nd.body;
            // Apply trim from neighboring tags' whitespace-control markers.
            if (i > 0 && nodes[i - 1].kind != Node::Text && nodes[i - 1].trim_right) {
                size_t a = t.find_first_not_of(" \t\r\n");
                t = (a == std::string::npos) ? "" : t.substr(a);
            }
            if (i + 1 < nodes.size() && nodes[i + 1].kind != Node::Text && nodes[i + 1].trim_left) {
                size_t b = t.find_last_not_of(" \t\r\n");
                t = (b == std::string::npos) ? "" : t.substr(0, b + 1);
            }
            out += t;
            ++i;
        } else if (nd.kind == Node::Output) {
            out += eval_string(nd.body, env);
            ++i;
        } else { // Stmt
            std::string kw = stmt_keyword(nd.body);
            if (kw == "if") {
                i = render_if(nodes, i, env, out);
            } else if (kw == "for") {
                i = render_for(nodes, i, env, out);
            } else {
                throw RenderError("unsupported statement: " + kw);
            }
        }
    }
}

} // namespace

// ============================================================================
// ChatTemplate
// ============================================================================

bool ChatTemplate::is_supported(const std::string& src) {
    // Reject templates using any construct outside the supported subset. Cheaper
    // and safer than partial rendering: if we see machinery we don't implement,
    // we fall back to the architecture default rather than emit wrong tokens.
    static const char* unsupported[] = {
        "macro", "namespace", "raise_exception", "tojson", "[::-1]",
        "tool", "vision", "image", "video", "set ", "elif",  // 'set'/'elif' deferred
        "loop.index ", "loop.last", "loop.previtem", "loop.nextitem",
        "reasoning_content", "selectattr", "%}\n{%- set",
    };
    for (const char* u : unsupported) {
        if (src.find(u) != std::string::npos) return false;
    }
    // Must contain a messages loop to be a usable chat template.
    return src.find("messages") != std::string::npos;
}

bool ChatTemplate::load(const std::string& model_dir, const std::string& fallback) {
    std::string located;

    // 1. chat_template.jinja
    std::string jinja_path = model_dir + "/chat_template.jinja";
    if (fs::exists(jinja_path)) {
        std::ifstream f(jinja_path, std::ios::binary);
        std::stringstream ss; ss << f.rdbuf();
        located = ss.str();
    }

    // 2. tokenizer_config.json -> "chat_template"
    if (located.empty()) {
        std::string tc_path = model_dir + "/tokenizer_config.json";
        std::ifstream f(tc_path, std::ios::binary);
        if (f.is_open()) {
            std::stringstream ss; ss << f.rdbuf();
            try {
                JsonValue root = JsonParser::parse(ss.str());
                if (root.contains("chat_template") && root["chat_template"].is_string()) {
                    located = root["chat_template"].str_val;
                }
            } catch (...) { /* ignore; fall through to fallback */ }
        }
    }

    // Use the located template only if it's within our supported subset;
    // otherwise fall back to the architecture's built-in default.
    if (!located.empty() && is_supported(located)) {
        template_src = located;
        have_template = true;
    } else if (!fallback.empty()) {
        template_src = fallback;
        have_template = true;
    } else {
        have_template = false;
    }
    return have_template;
}

bool ChatTemplate::render(const std::vector<ChatMessage>& messages,
                          bool add_generation_prompt, bool enable_thinking,
                          std::string& out) const {
    out.clear();
    if (!have_template) return false;
    try {
        std::vector<Node> nodes = tokenize(template_src);
        Env env;
        env.messages = &messages;
        env.add_generation_prompt = add_generation_prompt;
        env.enable_thinking = enable_thinking;
        render_range(nodes, 0, nodes.size(), env, out);
        return true;
    } catch (const std::exception& e) {
        if (std::getenv("QUICKLM_TPL_DEBUG")) {
            std::fprintf(stderr, "[chat_template] render error: %s\n", e.what());
        }
        out.clear();
        return false;
    }
}
