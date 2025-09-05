use anyhow::{anyhow, Result};
use regex::Regex;
use serde_json::{json, Value};

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
    let name_re = Regex::new(r"^[a-zA-Z0-9_.-]+$")?;
    if !name_re.is_match(name) {
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
                    let mut found = false;
                    for item in arr.iter() {
                        let s = item.as_str().ok_or_else(|| {
                            anyhow!("🐔 entries in 'parameters.type' array must be strings")
                        })?;
                        if s == "object" {
                            found = true;
                            break;
                        }
                    }
                    if !found {
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
        if let Some(props_val) = params.get("properties") {
            // Accept either an object or a JSON string encoding an object for properties.
            let props_map: serde_json::Map<String, Value> = if props_val.is_object() {
                props_val.as_object().unwrap().clone()
            } else if props_val.is_string() {
                let s = props_val.as_str().unwrap();
                let parsed = serde_json::from_str::<Value>(s)
                    .map_err(|_| anyhow!("🐔 'parameters.properties' string is not valid JSON"))?;
                if !parsed.is_object() {
                    return Err(anyhow!(
                        "🐔 'parameters.properties' string must decode to a JSON object"
                    ));
                }
                parsed.as_object().unwrap().clone()
            } else {
                return Err(anyhow!("🐔 'parameters.properties' must be an object or a JSON string encoding an object"));
            };
            let props = props_map;

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
                if !name_re.is_match(prop_name) {
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
                            let mut vec_types = Vec::new();
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
                                vec_types.push(s.to_string());
                            }
                            declared_types = Some(vec_types);
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

    Ok(Value::Object(out))
}

#[cfg(test)]
mod tests {
    use super::*;
    // use crate::common::*; // not used in these tests
    use serde::{Deserialize, Serialize};

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
    async fn test_render_tool_arguments_string() -> Result<()> {
        use serde_json::json;

        let v = json!({
            "name": "legacy",
            "arguments": "{ \"a\": 1 }",
            "extra": "should be dropped"
        });

        let out = normalize_tool(&v)?;
        let o = out.as_object().unwrap();
        assert!(o.get("extra").is_none());
        assert!(o.get("arguments").is_some());
        assert_eq!(o.get("arguments").unwrap()["a"], json!(1));

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
}
