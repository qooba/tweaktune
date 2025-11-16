use anyhow::{anyhow, Result};
use once_cell::sync::Lazy;
use regex::Regex;
use serde_json::{json, Value};
use std::collections::HashMap;

// Compile regex once and reuse across all validation calls
static NAME_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^[a-zA-Z0-9_.-]+$").expect("Failed to compile name regex"));

pub fn validate_function_call_format(value: &Value) -> Result<()> {
    // Accept both a "tool" definition (as in OpenAI function-calling tool schema)
    // and the older function-call object which may contain "arguments".
    // Validate top-level fields first.
    let obj = match value {
        Value::Object(m) => m,
        _ => return Err(anyhow!("🐔 function/tool definition must be a JSON object")),
    };

    // Validate "name"
    let name_val = obj
        .get("name")
        .ok_or_else(|| anyhow!("🐔 missing required field 'name'"))?;
    let name = name_val
        .as_str()
        .ok_or_else(|| anyhow!("🐔 'name' must be a string"))?;
    // Basic name character validation: letters, digits, underscore, dot or hyphen
    if !NAME_REGEX.is_match(name) {
        return Err(anyhow!("🐔 invalid function/tool name '{}'", name));
    }

    // Optional "description"
    if let Some(desc_val) = obj.get("description") {
        if !desc_val.is_string() {
            return Err(anyhow!("🐔 'description' must be a string if present"));
        }
    }

    if let Some(params_val) = obj.get("parameters") {
        if !params_val.is_object() {
            return Err(anyhow!(
                "🐔 'parameters' must be a JSON object (JSON Schema)"
            ));
        }
        let params = params_val.as_object().unwrap();

        // If `type` is present it should include "object". Allow string or array.
        if let Some(t) = params.get("type") {
            match t {
                Value::String(s) => {
                    if s != "object" {
                        return Err(anyhow!(
                            "🐔 'parameters.type' must be the string \"object\" when present"
                        ));
                    }
                }
                Value::Array(arr) => {
                    // Check all items are strings first
                    for item in arr.iter() {
                        if !item.is_string() {
                            return Err(anyhow!(
                                "🐔 entries in 'parameters.type' array must be strings"
                            ));
                        }
                    }
                    // Use iterator combinator for cleaner check
                    if !arr.iter().any(|item| item.as_str() == Some("object")) {
                        return Err(anyhow!(
                            "🐔 'parameters.type' array must include \"object\" when present"
                        ));
                    }
                }
                _ => {
                    return Err(anyhow!(
                        "🐔 'parameters.type' must be a string or an array of strings"
                    ));
                }
            }
        }

        // Validate `properties` if present. Some callers mistakenly place a `required`
        // array inside `properties` (instead of at the same level as `properties`).
        // Detect and extract that case so the `required` validation below can use it.
        let mut required_from_props: Option<Vec<String>> = None;

        // Parse properties string if needed (allocate only if string)
        let parsed_props: Option<Value>;

        if let Some(props_val) = params.get("properties") {
            // Accept either an object or a JSON string encoding an object for properties.
            let props = if props_val.is_object() {
                props_val.as_object().unwrap()
            } else if props_val.is_string() {
                let s = props_val.as_str().unwrap();
                parsed_props =
                    Some(serde_json::from_str::<Value>(s).map_err(|_| {
                        anyhow!("🐔 'parameters.properties' string is not valid JSON")
                    })?);
                if !parsed_props.as_ref().unwrap().is_object() {
                    return Err(anyhow!(
                        "🐔 'parameters.properties' string must decode to a JSON object"
                    ));
                }
                parsed_props.as_ref().unwrap().as_object().unwrap()
            } else {
                return Err(anyhow!("🐔 'parameters.properties' must be an object or a JSON string encoding an object"));
            };

            // If a `required` key was accidentally put inside `properties` and it's an array,
            // extract it and treat it as parameters.required.
            if let Some(req_in_props) = props.get("required") {
                if req_in_props.is_array() {
                    let mut vec = Vec::new();
                    for item in req_in_props.as_array().unwrap().iter() {
                        let s = item.as_str().ok_or_else(|| {
                            anyhow!("🐔 entries in misplaced 'properties.required' must be strings")
                        })?;
                        vec.push(s.to_string());
                    }
                    required_from_props = Some(vec);
                } else if req_in_props.is_null() {
                    // Treat a null `required` inside properties as an empty array
                    required_from_props = Some(Vec::new());
                }
            }

            for (prop_name, prop_schema) in props.iter() {
                // Skip the misplaced `required` entry we extracted above.
                if prop_name == "required" && required_from_props.is_some() {
                    continue;
                }

                // property names should be simple identifiers
                if !NAME_REGEX.is_match(prop_name) {
                    return Err(anyhow!(
                        "🐔 invalid parameter property name '{}' (only letters/digits/_ . - allowed)",
                        prop_name
                    ));
                }

                // each property's schema should be an object (a JSON Schema)
                if !prop_schema.is_object() {
                    return Err(anyhow!(
                        "🐔 schema for property '{}' must be a JSON object",
                        prop_name
                    ));
                }

                // If the property schema declares a `type`, it should be a string or array and one of
                // the common JSON Schema types. We do a conservative check only and allow "null".
                let mut declared_types: Option<Vec<String>> = None;
                if let Some(p_type) = prop_schema.get("type") {
                    let allowed = [
                        "string", "number", "integer", "boolean", "object", "array", "null",
                    ];
                    match p_type {
                        Value::String(s) => {
                            if !allowed.contains(&s.as_str()) {
                                return Err(anyhow!(
                                    "🐔 unsupported 'type' '{}' for property '{}'",
                                    s,
                                    prop_name
                                ));
                            }
                            declared_types = Some(vec![s.clone()]);
                        }
                        Value::Array(arr) => {
                            // Validate all types first
                            for item in arr.iter() {
                                let s = item.as_str().ok_or_else(|| {
                                    anyhow!("🐔 entries in property 'type' array must be strings")
                                })?;
                                if !allowed.contains(&s) {
                                    return Err(anyhow!(
                                        "🐔 unsupported 'type' '{}' for property '{}'",
                                        s,
                                        prop_name
                                    ));
                                }
                            }
                            // Collect using iterator for efficiency
                            declared_types = Some(
                                arr.iter()
                                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                                    .collect(),
                            );
                        }
                        _ => {
                            return Err(anyhow!(
                                "🐔 'type' for property '{}' must be a string or array of strings",
                                prop_name
                            ));
                        }
                    }
                }

                // If an `enum` is present, it should be an array and each enum item's JSON type
                // should be compatible with the declared `type` (if any). We allow nulls when
                // "null" is declared in the property's type(s).
                if let Some(enum_val) = prop_schema.get("enum") {
                    if !enum_val.is_array() {
                        return Err(anyhow!(
                            "🐔 'enum' for property '{}' must be an array",
                            prop_name
                        ));
                    }

                    for item in enum_val.as_array().unwrap().iter() {
                        // determine runtime type string of the enum item
                        let item_type = match item {
                            Value::String(_) => "string",
                            Value::Bool(_) => "boolean",
                            Value::Number(n) => {
                                if n.is_i64() || n.is_u64() {
                                    "integer"
                                } else {
                                    "number"
                                }
                            }
                            Value::Object(_) => "object",
                            Value::Array(_) => "array",
                            Value::Null => "null",
                        };

                        if let Some(dtypes) = &declared_types {
                            // If any declared type matches the item type, accept it. Note that
                            // "number" accepts both integer and number JSON numbers, but
                            // if declared is "integer" and item is non-integer, it's invalid.
                            let mut ok = false;
                            for dt in dtypes.iter() {
                                if dt == item_type {
                                    ok = true;
                                    break;
                                }
                                if dt == "number" && item_type == "integer" {
                                    ok = true;
                                    break;
                                }
                                if dt == "integer" && item_type == "number" {
                                    // item is a float but declared is integer -> not ok
                                    continue;
                                }
                            }
                            if !ok {
                                return Err(anyhow!(
                                    "🐔 enum value for property '{}' has type '{}' which is not compatible with declared types {:?}",
                                    prop_name,
                                    item_type,
                                    dtypes
                                ));
                            }
                        }
                    }
                }
            }
        }

        // Validate `required` if present: must be array of strings and subset of properties
        let mut required_list: Option<Vec<String>> = None;
        if let Some(req_val) = params.get("required") {
            // Treat null as an empty array; otherwise ensure it's an array.
            let req_val = if req_val.is_null() {
                Value::Array(Vec::new())
            } else if req_val.is_array() {
                req_val.clone()
            } else {
                return Err(anyhow!(
                    "🐔 'parameters.required' must be an array of strings"
                ));
            };
            let mut vec = Vec::new();
            for item in req_val.as_array().unwrap().iter() {
                let s = item.as_str().ok_or_else(|| {
                    anyhow!("🐔 entries in 'parameters.required' must be strings")
                })?;
                vec.push(s.to_string());
            }
            required_list = Some(vec);
        } else if let Some(rp) = required_from_props {
            required_list = Some(rp);
        }

        if let Some(req_arr) = &required_list {
            // If properties exist, collect their names for subset check
            let prop_keys: Option<std::collections::HashSet<String>> = params
                .get("properties")
                .and_then(|p| p.as_object())
                .map(|m| m.keys().cloned().collect());

            for s in req_arr.iter() {
                if let Some(keys) = &prop_keys {
                    if !keys.contains(s) {
                        return Err(anyhow!(
                            "🐔 'parameters.required' contains '{}' which is not defined in properties",
                            s
                        ));
                    }
                }
            }
        }
    } else {
        // Backwards compatible validation for function-call-style objects that use
        // an "arguments" field (actual call payload) rather than a JSON Schema.
        if let Some(args_val) = obj.get("arguments") {
            match args_val {
                Value::Object(_) | Value::Null => {
                    // ok
                }
                Value::String(s) => {
                    // allow arguments as a JSON string containing an object
                    let parsed = serde_json::from_str::<Value>(s)
                        .map_err(|_| anyhow!("🐔 'arguments' string is not valid JSON"))?;
                    if !parsed.is_object() {
                        return Err(anyhow!(
                            "🐔 'arguments' string must decode to a JSON object"
                        ));
                    }
                }
                _ => {
                    return Err(anyhow!(
                        "🐔 'arguments' must be a JSON object, null, or a JSON string encoding an object"
                    ));
                }
            }
        }
    }

    Ok(())
}

pub fn normalize_tool(value: &Value) -> Result<Value> {
    let obj = match value {
        Value::Object(m) => m,
        _ => return Err(anyhow!("🐔 function/tool must be a JSON object")),
    };

    // name is required and must be string
    let name_val = obj
        .get("name")
        .ok_or_else(|| anyhow!("🐔 missing required field 'name'"))?;
    let name = name_val
        .as_str()
        .ok_or_else(|| anyhow!("🐔 'name' must be a string"))?;

    // Start building canonical object
    let mut out = serde_json::Map::new();
    out.insert("name".to_string(), Value::String(name.to_string()));

    // optional description
    if let Some(d) = obj.get("description") {
        if d.is_string() {
            out.insert(
                "description".to_string(),
                Value::String(d.as_str().unwrap().to_string()),
            );
        }
    }

    // If parameters present and is object or string-encoded object, keep it as object
    if let Some(params) = obj.get("parameters") {
        if params.is_object() {
            if let Some(props) = params.get("properties") {
                if props.is_string() {
                    // If properties is a string, try to parse it as JSON
                    let parsed =
                        serde_json::from_str::<Value>(props.as_str().unwrap()).map_err(|_| {
                            anyhow!("🐔 'parameters.properties' string is not valid JSON")
                        })?;
                    if !parsed.is_object() {
                        return Err(anyhow!(
                            "🐔 'parameters.properties' string must decode to a JSON object"
                        ));
                    }
                    let mut new_params = params.clone();

                    if let Some(o) = new_params.as_object_mut() {
                        o.insert("properties".to_string(), parsed.clone());
                    }

                    out.insert("parameters".to_string(), new_params);
                } else {
                    out.insert("parameters".to_string(), params.clone());
                }
            } else {
                out.insert("parameters".to_string(), params.clone());
            }
        } else if params.is_string() {
            if let Ok(parsed) = serde_json::from_str::<Value>(params.as_str().unwrap()) {
                if parsed.is_object() {
                    out.insert("parameters".to_string(), parsed);
                }
            }
        }
    }

    if let Some(Value::Object(o)) = out.get_mut("parameters") {
        if let Some(Value::Object(props)) = o.get_mut("properties") {
            if let Some(req) = props.remove("required") {
                if o.get("required").is_none() {
                    o.insert("required".to_string(), req);
                }
            }
        } else if o.get("required").is_none() {
            o.insert("required".to_string(), Value::Array(Vec::new()));
        }
    }

    Ok(Value::Object(out))
}

pub fn validate_function_call_conversation(value: &Value) -> Result<()> {
    let obj = match value {
        Value::Object(m) => m,
        _ => return Err(anyhow!("🐔 conversation root must be a JSON object")),
    };

    // Validate optional function_descriptions first so we can reference them
    let mut known_funcs: std::collections::HashSet<String> = std::collections::HashSet::new();
    if let Some(fd) = obj.get("function_descriptions") {
        if !fd.is_array() {
            return Err(anyhow!(
                "🐔 'function_descriptions' must be an array when present"
            ));
        }
        for item in fd.as_array().unwrap().iter() {
            // reuse existing validator which accepts tool/function definitions
            validate_function_call_format(item)?;
            if let Some(name) = item.get("name").and_then(|v| v.as_str()) {
                known_funcs.insert(name.to_string());
            }
        }
    }

    // conversation must be an array
    let conv = obj
        .get("conversation")
        .ok_or_else(|| anyhow!("🐔 missing required field 'conversation'"))?;
    if !conv.is_array() {
        return Err(anyhow!("🐔 'conversation' must be an array"));
    }
    for (idx, entry) in conv.as_array().unwrap().iter().enumerate() {
        if !entry.is_object() {
            return Err(anyhow!("🐔 conversation[{}] must be an object", idx));
        }
        let e = entry.as_object().unwrap();

        // speaker
        let speaker = e
            .get("speaker")
            .ok_or_else(|| anyhow!("🐔 conversation[{}] missing 'speaker'", idx))?;
        let speaker_s = speaker
            .as_str()
            .ok_or_else(|| anyhow!("🐔 conversation[{}].speaker must be a string", idx))?;
        if speaker_s != "human" && speaker_s != "assistant" && speaker_s != "system" {
            return Err(anyhow!(
                "🐔 conversation[{}].speaker must be 'human', 'assistant', or 'system'",
                idx
            ));
        }

        // message may be null or string
        if let Some(msg) = e.get("message") {
            if !(msg.is_null() || msg.is_string()) {
                return Err(anyhow!(
                    "🐔 conversation[{}].message must be null or a string",
                    idx
                ));
            }
        } else {
            return Err(anyhow!("🐔 conversation[{}] missing 'message'", idx));
        }

        // action may be null or string
        let action = e.get("action");
        if action.is_none() {
            return Err(anyhow!("🐔 conversation[{}] missing 'action'", idx));
        }
        let action = action.unwrap();
        if !(action.is_null() || action.is_string()) {
            return Err(anyhow!(
                "🐔 conversation[{}].action must be null or a string",
                idx
            ));
        }

        // details may be null or object
        if let Some(details) = e.get("details") {
            if !(details.is_null() || details.is_object()) {
                return Err(anyhow!(
                    "🐔 conversation[{}].details must be null or an object",
                    idx
                ));
            }

            // If this is a function-call action, validate structure
            if action.is_string() && action.as_str().unwrap() == "function-call" {
                if details.is_null() {
                    return Err(anyhow!(
                        "🐔 conversation[{}] function-call must have non-null details",
                        idx
                    ));
                }
                let d = details.as_object().unwrap();
                // expect name and arguments inside details
                let name_val = d.get("name").ok_or_else(|| {
                    anyhow!(
                        "🐔 conversation[{}].details must contain 'name' for function-call",
                        idx
                    )
                })?;
                let name = name_val
                    .as_str()
                    .ok_or_else(|| anyhow!("🐔 conversation details.name must be a string"))?;

                // optional: check referenced function exists in description list
                if !known_funcs.is_empty() && !known_funcs.contains(name) {
                    return Err(anyhow!(
                        "🐔 conversation[{}] references unknown function '{}'",
                        idx,
                        name
                    ));
                }

                let args_val = d.get("arguments").ok_or_else(|| {
                    anyhow!(
                        "🐔 conversation[{}].details must contain 'arguments' for function-call",
                        idx
                    )
                })?;

                match args_val {
                    Value::Object(_) | Value::Null => {}
                    Value::String(s) => {
                        let parsed = serde_json::from_str::<Value>(s).map_err(|_| {
                            anyhow!(
                                "🐔 conversation[{}].details.arguments string is not valid JSON",
                                idx
                            )
                        })?;
                        if !parsed.is_object() {
                            return Err(anyhow!(
                                "🐔 conversation[{}].details.arguments must decode to an object",
                                idx
                            ));
                        }
                    }
                    _ => {
                        return Err(anyhow!("🐔 conversation[{}].details.arguments must be object, null or JSON string", idx));
                    }
                }
            }

            // If this is a function-response action, ensure details is object mapping function name -> result
            if action.is_string() && action.as_str().unwrap() == "function-response" {
                if details.is_null() {
                    return Err(anyhow!(
                        "🐔 conversation[{}] function-response must have non-null details",
                        idx
                    ));
                }
                if let Some(map) = details.as_object() {
                    if map.len() != 1 {
                        // allow single-key mapping where key is function name
                        // but allow multiple in case of aggregated responses
                    }
                    for (k, v) in map.iter() {
                        // key should be a simple name
                        if !NAME_REGEX.is_match(k) {
                            return Err(anyhow!("🐔 conversation[{}] function-response has invalid function name '{}'", idx, k));
                        }
                        // value may be object or primitive
                        if !(v.is_null()
                            || v.is_object()
                            || v.is_string()
                            || v.is_number()
                            || v.is_boolean())
                        {
                            return Err(anyhow!("🐔 conversation[{}] function-response value for '{}' must be JSON value", idx, k));
                        }
                    }
                } else {
                    return Err(anyhow!(
                        "🐔 conversation[{}].details for function-response must be an object",
                        idx
                    ));
                }
            }
        } else {
            return Err(anyhow!("🐔 conversation[{}] missing 'details'", idx));
        }
    }

    Ok(())
}

/// Validate alternate conversation format that uses `role`, `content`, `tool_calls`, and `tools`.
/// `value` is a JSON string for convenience (many callers produce string payloads).
pub fn validate_tool_format_messages(value: &Value) -> Result<()> {
    let obj = match value {
        Value::Object(m) => m,
        _ => return Err(anyhow!("🐔 messages root must be a JSON object")),
    };

    // Validate optional `tools` array. If `tools` key is present (even if empty),
    // enforce that assistant tool_calls reference only names from this list.
    // Use HashMap for O(1) tool lookups instead of O(n) linear search
    let mut known_tools: HashMap<&str, &Value> = HashMap::new();
    let tools_provided = obj.get("tools").is_some();
    if let Some(tools) = obj.get("tools") {
        if !tools.is_array() {
            return Err(anyhow!("🐔 'tools' must be an array when present"));
        }
        for t in tools.as_array().unwrap().iter() {
            // reuse existing validation for tool schema
            validate_function_call_format(t)?;
            if let Some(n) = t.get("name").and_then(|v| v.as_str()) {
                known_tools.insert(n, t);
            }
        }
    }

    // messages must be an array
    let conv = obj
        .get("messages")
        .ok_or_else(|| anyhow!("🐔 missing required field 'messages'"))?;
    if !conv.is_array() {
        return Err(anyhow!("🐔 'messages' must be an array"));
    }
    for (idx, entry) in conv.as_array().unwrap().iter().enumerate() {
        if !entry.is_object() {
            return Err(anyhow!("🐔 messages[{}] must be an object", idx));
        }
        let e = entry.as_object().unwrap();

        // role required and must be string (user/assistant/tool/system)
        let role = e
            .get("role")
            .ok_or_else(|| anyhow!("🐔 messages[{}] missing 'role'", idx))?;
        let role_s = role
            .as_str()
            .ok_or_else(|| anyhow!("🐔 messages[{}].role must be a string", idx))?;
        if role_s != "user" && role_s != "assistant" && role_s != "tool" && role_s != "system" {
            return Err(anyhow!(
                "🐔 messages[{}].role must be 'user','assistant','tool' or 'system'",
                idx
            ));
        }

        // content must be present for user/tool; assistant may omit content if it has tool_calls
        if role_s == "user" || role_s == "tool" {
            let content = e
                .get("content")
                .ok_or_else(|| anyhow!("🐔 messages[{}] missing 'content'", idx))?;
            if !content.is_string() {
                return Err(anyhow!("🐔 messages[{}].content must be a string", idx));
            }
        }

        // If assistant entry has tool_calls, validate array of objects with function/name/arguments.
        // Assistant entries are valid if they have either `content` (string) or `tool_calls`.
        if role_s == "assistant" {
            let has_content = e.get("content").is_some();
            let has_tool_calls = e.get("tool_calls").is_some();
            let has_think = e.get("reasoning_content").is_some();
            if !has_content && !has_tool_calls && !has_think {
                return Err(anyhow!(
                    "🐔 messages[{}] assistant entry must have 'content' or 'tool_calls' or 'reasoning_content'",
                    idx
                ));
            }

            if let Some(tc) = e.get("tool_calls") {
                if !tc.is_array() {
                    return Err(anyhow!(
                        "🐔 conversation[{}].tool_calls must be an array",
                        idx
                    ));
                }
                for (j, call) in tc.as_array().unwrap().iter().enumerate() {
                    if !call.is_object() {
                        return Err(anyhow!(
                            "🐔 messages[{}].tool_calls[{}] must be an object",
                            idx,
                            j
                        ));
                    }
                    let call_obj = call.as_object().unwrap();
                    let function = call_obj.get("function").ok_or_else(|| {
                        anyhow!(
                            "🐔 conversation[{}].tool_calls[{}] missing 'function'",
                            idx,
                            j
                        )
                    })?;
                    if !function.is_object() {
                        return Err(anyhow!(
                            "🐔 conversation[{}].tool_calls[{}].function must be an object",
                            idx,
                            j
                        ));
                    }
                    let func_obj = function.as_object().unwrap();
                    let name = func_obj
                        .get("name")
                        .ok_or_else(|| {
                            anyhow!(
                                "🐔 messages[{}].tool_calls[{}].function missing 'name'",
                                idx,
                                j
                            )
                        })?
                        .as_str()
                        .ok_or_else(|| {
                            anyhow!("🐔 messages[].tool_calls[].function.name must be a string")
                        })?;
                    if !NAME_REGEX.is_match(name) {
                        return Err(anyhow!(
                            "🐔 messages[{}].tool_calls[{}] invalid function name '{}'",
                            idx,
                            j,
                            name
                        ));
                    }
                    if tools_provided && !known_tools.contains_key(name) {
                        return Err(anyhow!(
                            "🐔 messages[{}].tool_calls[{}] references unknown tool '{}'",
                            idx,
                            j,
                            name
                        ));
                    }

                    // arguments may be object or string
                    if let Some(args) = func_obj.get("arguments") {
                        match args {
                            Value::Object(_) | Value::Null => {}
                            _ => {
                                return Err(anyhow!("🐔 messages[{}].tool_calls[{}].function.arguments must be object or null", idx, j));
                            }
                        }

                        // O(1) HashMap lookup instead of O(n) linear search
                        if tools_provided {
                            if let Some(used_tool_schema) = known_tools.get(name) {
                                let properties = if let Value::String(ref v) =
                                    used_tool_schema["parameters"]["properties"]
                                {
                                    serde_json::from_str(v).unwrap()
                                } else {
                                    used_tool_schema["parameters"]["properties"].clone()
                                };

                                let schema_value = json!({
                                    "type": "object",
                                    "properties": properties,
                                    "required": used_tool_schema["parameters"]["required"],
                                    "additionalProperties": used_tool_schema["parameters"]["additionalProperties"].as_bool().unwrap_or(false),
                                });

                                let is_valid = jsonschema::is_valid(&schema_value, args);

                                if !is_valid {
                                    return Err(anyhow!(
                                        "🐔 messages[{}].tool_calls[{}].function.arguments does not conform to tool schema",
                                        idx,
                                        j
                                    ));
                                }
                            }
                        }
                    } else {
                        return Err(anyhow!(
                            "🐔 messages[{}].tool_calls[{}].function missing 'arguments'",
                            idx,
                            j
                        ));
                    }
                }
            }
        }

        // If role == tool, content is usually JSON string; ensure it's valid JSON and object
        if role_s == "tool" {
            if let Some(content) = e.get("content") {
                if let Some(s) = content.as_str() {
                    let parsed = serde_json::from_str::<Value>(s).map_err(|_| {
                        anyhow!(
                            "🐔 messages[{}].content for tool must be a valid JSON string",
                            idx
                        )
                    })?;
                    if !parsed.is_object() {
                        return Err(anyhow!(
                            "🐔 messages[{}].content for tool must decode to an object",
                            idx
                        ));
                    }
                }
            }
        }
    }

    Ok(())
}

/*
pub fn validate_tool_call_schema(value: &Value) -> Result<()> {
    let schema_value = json!({
        "type": "object",
        "properties": properties,
        "required": full_schema["required"],
        "additionalProperties": false,
    });

    let instance_json = templates.render(self.instance.clone(), context.data.clone())?;

    match serde_json::from_str(&instance_json) {
        Ok(instance) => {
            let is_valid = jsonschema::is_valid(&schema_value, &instance);

            if !is_valid {
                error!(target: "validate_json_step", "🐔 Failed to validate JSON: {} with schema {}", instance, schema_value);
                context.set_status(StepStatus::Failed);
            }

            Ok(context)
        }
        Err(e) => {
            error!(target: "validate_json_step", "🐔 Failed to render instance: {}", e);
            error!(target: "validate_json_step", "🐔 INSTANCE_JSON: {}", &instance_json);
            context.set_status(StepStatus::Failed);
            Ok(context)
        }
    }
}
    */

#[cfg(test)]
mod tests {
    use super::*;
    // use crate::common::*; // not used in these tests
    use serde::{Deserialize, Serialize};
    use serde_json::json;

    #[derive(Serialize, Deserialize, Debug, Clone)]
    struct Person {
        name: String,
    }

    #[tokio::test]
    async fn test_validate_function_call_format_tool_valid() -> Result<()> {
        use serde_json::json;

        let v = json!({
            "name": "search_books",
            "description": "Search for books by title or author",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": { "type": "string" },
                    "year": { "type": "integer" }
                },
                "required": ["title"]
            }
        });

        validate_function_call_format(&v)?;

        Ok(())
    }

    #[tokio::test]
    async fn test_validate_function_call_format_invalid_name() -> Result<()> {
        use serde_json::json;

        let v = json!({ "name": "bad name!" });

        let res = validate_function_call_format(&v);
        assert!(res.is_err());

        Ok(())
    }

    #[tokio::test]
    async fn test_validate_function_call_format_legacy_arguments_string() -> Result<()> {
        use serde_json::json;

        let v = json!({
            "name": "legacy_call",
            "arguments": "{ \"a\": 1, \"b\": \"x\" }"
        });

        validate_function_call_format(&v)?;

        Ok(())
    }

    #[tokio::test]
    async fn test_validate_function_call_format_parameters_required_missing() -> Result<()> {
        use serde_json::json;

        let v = json!({
            "name": "broken",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": { "type": "string" }
                },
                "required": ["b"]
            }
        });

        let res = validate_function_call_format(&v);
        assert!(res.is_err());

        Ok(())
    }

    #[tokio::test]
    async fn test_enum_string_valid() -> Result<()> {
        use serde_json::json;

        let v = json!({
            "name": "enum_string",
            "parameters": {
                "type": "object",
                "properties": {
                    "color": { "type": "string", "enum": ["red", "green"] }
                },
                "required": ["color"]
            }
        });

        validate_function_call_format(&v)?;
        Ok(())
    }

    #[tokio::test]
    async fn test_enum_integer_valid() -> Result<()> {
        use serde_json::json;

        let v = json!({
            "name": "enum_integer",
            "parameters": {
                "type": "object",
                "properties": {
                    "level": { "type": "integer", "enum": [1, 2, 3] }
                }
            }
        });

        validate_function_call_format(&v)?;
        Ok(())
    }

    #[tokio::test]
    async fn test_enum_number_with_integer_valid() -> Result<()> {
        use serde_json::json;

        let v = json!({
            "name": "enum_number",
            "parameters": {
                "type": "object",
                "properties": {
                    "ratio": { "type": "number", "enum": [1, 2.5] }
                }
            }
        });

        validate_function_call_format(&v)?;
        Ok(())
    }

    #[tokio::test]
    async fn test_render_tool_basic() -> Result<()> {
        use serde_json::json;

        let v = json!({
            "name": "do_thing",
            "description": "Does a thing",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": { "type": "integer" }
                }
            }
        });

        let out = normalize_tool(&v)?;
        assert!(out.is_object());
        let o = out.as_object().unwrap();
        assert_eq!(o.get("name").and_then(|v| v.as_str()), Some("do_thing"));
        assert!(o.get("parameters").is_some());

        Ok(())
    }

    #[tokio::test]
    async fn test_enum_integer_with_float_invalid() -> Result<()> {
        use serde_json::json;

        let v = json!({
            "name": "enum_bad_integer",
            "parameters": {
                "type": "object",
                "properties": {
                    "level": { "type": "integer", "enum": [1, 2, 2.5] }
                }
            }
        });

        let res = validate_function_call_format(&v);
        assert!(res.is_err());
        Ok(())
    }

    #[tokio::test]
    async fn test_enum_null_allowed() -> Result<()> {
        use serde_json::json;

        let v = json!({
            "name": "enum_null",
            "parameters": {
                "type": "object",
                "properties": {
                    "maybe": { "type": ["string", "null"], "enum": ["x", null] }
                }
            }
        });

        validate_function_call_format(&v)?;
        Ok(())
    }

    #[tokio::test]
    async fn test_enum_null_not_allowed_invalid() -> Result<()> {
        use serde_json::json;

        let v = json!({
            "name": "enum_null_bad",
            "parameters": {
                "type": "object",
                "properties": {
                    "maybe": { "type": "string", "enum": [null] }
                }
            }
        });

        let res = validate_function_call_format(&v);
        assert!(res.is_err());
        Ok(())
    }

    #[tokio::test]
    async fn test_validate_function_call_conversation_valid() -> Result<()> {
        use serde_json::json;

        let v = json!({
            "conversation": [
                {
                    "speaker": "human",
                    "message": "Ile wyniesie indywidualna kwota napiwku...",
                    "action": null,
                    "details": null
                },
                {
                    "speaker": "assistant",
                    "message": null,
                    "action": "function-call",
                    "details": {
                        "name": "calculate_tip_split",
                        "arguments": { "total_bill": 250, "number_of_people": 5 }
                    }
                },
                {
                    "speaker": "assistant",
                    "message": null,
                    "action": "function-response",
                    "details": { "calculate_tip_split": { "individual_tip_amount": 50 } }
                },
                {
                    "speaker": "assistant",
                    "message": "Indywidualna kwota napiwku...",
                    "action": null,
                    "details": null
                }
            ],
            "function_descriptions": [
                {
                    "name": "calculate_tip_split",
                    "description": "Obliczanie indywidualnej kwoty napiwku dla grupy",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "total_bill": { "type": "number" },
                            "number_of_people": { "type": "integer" }
                        },
                        "required": ["total_bill","number_of_people"]
                    }
                }
            ]
        });

        validate_function_call_conversation(&v)?;
        Ok(())
    }

    #[tokio::test]
    async fn test_validate_function_call_conversation_missing_details() -> Result<()> {
        use serde_json::json;

        let v = json!({
            "conversation": [
                {
                    "speaker": "assistant",
                    "message": null,
                    "action": "function-call",
                    "details": null
                }
            ]
        });

        let res = validate_function_call_conversation(&v);
        assert!(res.is_err());
        Ok(())
    }

    #[tokio::test]
    async fn test_validate_tool_format_conversation_valid() -> Result<()> {
        let s = json!(
        {
            "messages": [
                        { "role": "user", "content": "Ile wyniesie indywidualna kwota napiwku...?" },
                        { "role": "assistant", "tool_calls": [
                                { "function": { "name": "calculate_tip_split", "arguments": { "total_bill": 250, "number_of_people": 5 } } },
                                { "function": { "name": "calculate_tip_split", "arguments": { "total_bill": 250, "number_of_people": 5 } } }
                        ] },
                        { "role": "tool", "content": "{\"calculate_tip_split\": {\"individual_tip_amount\": 50}}" },
                        { "role": "assistant", "content": "Indywidualna kwota napiwku..." }
                    ],
                    "tools": [
                        {
                            "name": "calculate_tip_split",
                            "description": "Oblicza indywidualną kwotę napiwku",
                            "parameters": { "type": "object", "properties": { "total_bill": { "type": "number" }, "number_of_people": { "type": "integer" } }, "required": ["total_bill","number_of_people"] }
                        }
                    ]
                }
            );

        validate_tool_format_messages(&s)?;
        Ok(())
    }

    #[tokio::test]
    async fn test_validate_tool_format_conversation_invalid_unknown_tool() -> Result<()> {
        let s = json!(
            {
                "messages": [
                    { "role": "assistant", "tool_calls": [ { "function": { "name": "unknown_tool", "arguments": { } } } ] }
                ],
                "tools": []
            }
        );

        let res = validate_tool_format_messages(&s);
        assert!(res.is_err());
        Ok(())
    }
}
