from compiler.dsl import DSLCompiler

def test_dsl_compiler_output():
    compiler = DSLCompiler()
    res = compiler.compile("SCENE main { OBJECT chair_0; }")
    assert res["status"] == "compiled"
    assert "ast" in res
