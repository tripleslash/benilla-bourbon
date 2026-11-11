//! pfUI carry: the Lua 5.1 expressions a pfUI state takes (`Heap::set_pfui_lua51`), and a
//! stock state's grammar left as the 1.12 client's.

fn run_on(lua51: bool, src: &str) -> Result<Vec<String>, String> {
    let mut vm = benilla_lua::new_state();
    vm.heap.set_pfui_lua51(lua51);
    match vm.exec(src.as_bytes(), b"=t") {
        Ok(vs) => Ok(vs.into_iter().map(|v| vm.display(v)).collect()),
        Err(e) => Err(vm.display(e)),
    }
}

fn ok(src: &str) -> Vec<String> {
    run_on(true, src).unwrap_or_else(|e| panic!("{src}: {e}"))
}

#[test]
fn modulo_takes_the_divisors_sign_as_5_1_does() {
    assert_eq!(
        ok("return 7 % 3, -7 % 3, 7 % -3, 5.5 % 2"),
        ["1", "2", "-2", "1.5"]
    );
    assert_eq!(
        ok("local a, b = 10, 4 return a % b * 2, 1 + 9 % 4"),
        ["4", "2"]
    );
    assert_eq!(ok("return '9' % 4"), ["1"]);
    assert!(run_on(true, "return {} % 2")
        .unwrap_err()
        .contains("perform arithmetic on"));
}

#[test]
fn length_reads_a_strings_bytes_and_a_tables_border() {
    assert_eq!(
        ok("local t = {1, 2, 3} return #t, #'abcd', #{}"),
        ["3", "4", "0"]
    );
    assert_eq!(ok("local s = 'xy' return #s + 1, -#s"), ["3", "-2"]);
    assert!(run_on(true, "return #5")
        .unwrap_err()
        .contains("get length of"));
}

#[test]
fn dots_are_the_varargs_in_a_function_and_the_chunks_arguments() {
    assert_eq!(
        ok("local function f(...) return ... end return f(1, nil, 3)"),
        ["1", "nil", "3"]
    );
    assert_eq!(
        ok("local function f(a, ...) local t = {...} return a, table.getn(t) end return f(9, 1, 2)"),
        ["9", "2"]
    );
    assert_eq!(ok("local x, y = ... return x, y"), ["nil", "nil"]);
    // A vararg function forwarding its own `...` through an indexed callee in a closure, as
    // pfUI's ID companions do.
    assert_eq!(
        ok("local src = { add = function(a, b) return a + b end }
            local fns = {}
            for name, source in pairs({ plus = 'add' }) do
              fns[name] = function(...) return src[source](...) end
            end
            return fns.plus(2, 5)"),
        ["7"]
    );
    assert!(run_on(true, "local function f() return ... end")
        .unwrap_err()
        .contains("outside a vararg function"));
}

#[test]
fn a_stock_state_keeps_the_1_12_grammar() {
    for src in [
        "return 7 % 3",
        "return #'abc'",
        "local function f(...) return ... end",
    ] {
        assert!(
            run_on(false, src).is_err(),
            "{src} compiled on a stock state"
        );
    }
    // `arg` at chunk level stays the global it is in 5.0, on either state.
    assert_eq!(run_on(false, "arg = 4 return arg").unwrap(), ["4"]);
    assert_eq!(ok("arg = 4 return arg"), ["4"]);
}

#[test]
fn each_pass_of_a_loop_has_its_own_variables() {
    // 5.1: a closure made in the body keeps its pass's values, in a generic and a numeric for.
    assert_eq!(
        ok("local fns = {}
            for name, v in pairs({ a = 1 }) do fns[name] = function() return v end end
            for i = 1, 3 do fns[i] = function() return i end end
            return fns.a(), fns[1](), fns[3]()"),
        ["1", "1", "3"]
    );
    // A break out of a pass that captured its variables closes them too.
    assert_eq!(
        ok(
            "local f for i = 1, 5 do f = function() return i end if i == 2 then break end end
            return f()"
        ),
        ["2"]
    );
    // 5.0 shares one set across the loop; a stock state keeps that.
    assert_eq!(
        run_on(
            false,
            "local f for i = 1, 3 do if i == 1 then f = function() return i end end end return f()"
        )
        .unwrap()
        .len(),
        1
    );
}
