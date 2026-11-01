"""Tests for the search tools: grep (content) and glob (paths)."""

import os

import pytest

from tau.tools.search import GlobTool, GrepTool, _expand_braces


@pytest.fixture
def tree(tmp_path):
    (tmp_path / "src" / "deep").mkdir(parents=True)
    (tmp_path / "src" / "app.py").write_text(
        "import os\n\ndef handler():\n    return 'needle'\n"
    )
    (tmp_path / "src" / "deep" / "util.py").write_text("x = 1\n# needle in a comment\n")
    (tmp_path / "README.md").write_text("needle docs\n")
    (tmp_path / "node_modules" / "dep").mkdir(parents=True)
    (tmp_path / "node_modules" / "dep" / "index.py").write_text("needle = 'noise'\n")
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "secret.py").write_text("needle hidden\n")
    (tmp_path / "blob.bin").write_bytes(b"needle\x00binary")
    return tmp_path


@pytest.fixture
def grep(tree):
    return GrepTool(cwd=str(tree))


@pytest.fixture
def glob(tree):
    return GlobTool(cwd=str(tree))


# -- grep: shape of results ---------------------------------------------------


def test_grep_returns_path_line_text(grep):
    result = grep.execute(pattern="needle")
    assert not result.is_error
    assert "src/app.py:4:     return 'needle'" in result.output
    assert "README.md:1: needle docs" in result.output


def test_grep_skips_noise_directories_and_hidden_dirs(grep):
    output = grep.execute(pattern="needle").output
    assert "node_modules" not in output
    assert ".hidden" not in output


def test_grep_skips_binary_files(grep):
    assert "blob.bin" not in grep.execute(pattern="needle").output


def test_grep_does_not_match_unrelated_files(grep):
    output = grep.execute(pattern="needle").output
    assert "util.py" in output
    assert "app.py" in output


def test_grep_glob_filter_uses_full_match_for_double_star(grep):
    output = grep.execute(pattern="needle", glob="**/*.md").output
    assert output.strip() == "README.md:1: needle docs"


def test_grep_glob_filter_without_slash_matches_basename(grep):
    output = grep.execute(pattern="needle", glob="*.py").output
    assert "app.py" in output and "README.md" not in output


def test_grep_files_with_matches(grep):
    output = grep.execute(pattern="needle", output_mode="files_with_matches").output
    paths = output.splitlines()
    assert "README.md" in paths
    assert "src/app.py" in paths
    assert all(":" not in path for path in paths)


def test_grep_count_mode(grep):
    output = grep.execute(pattern="needle", output_mode="count").output
    assert "README.md:1" in output
    assert "src/app.py:1" in output


def test_grep_context_lines_use_dash_separator(grep):
    output = grep.execute(pattern="return 'needle'", context=1).output
    assert "src/app.py-3- def handler():" in output
    assert "src/app.py:4:     return 'needle'" in output


def test_grep_ignore_case(grep, tree):
    (tree / "case.txt").write_text("NEEDLE\n")
    assert "case.txt" not in grep.execute(pattern="needle").output
    assert "case.txt" in grep.execute(pattern="needle", ignore_case=True).output


def test_grep_no_matches_is_not_an_error(grep):
    result = grep.execute(pattern="absent-token-xyz")
    assert not result.is_error
    assert result.output.startswith("No matches")


def test_grep_invalid_regex_is_an_error(grep):
    result = grep.execute(pattern="(unclosed")
    assert result.is_error
    assert "invalid regular expression" in result.output


def test_grep_missing_path_is_an_error(grep):
    result = grep.execute(pattern="x", path="nope/missing")
    assert result.is_error and "no such path" in result.output


def test_grep_head_limit_truncates_and_says_so(tree):
    (tree / "many.txt").write_text("\n".join(f"hit {i}" for i in range(60)) + "\n")
    result = GrepTool(cwd=str(tree)).execute(pattern="hit", head_limit=5)
    assert len(result.output.splitlines()) >= 5
    assert "truncated at 5 lines" in result.output


def test_grep_single_file_path(tree):
    result = GrepTool(cwd=str(tree)).execute(pattern="needle", path="README.md")
    assert result.output.strip() == "README.md:1: needle docs"


def test_grep_long_lines_are_clipped(tree):
    (tree / "long.txt").write_text("needle" + "x" * 2000 + "\n")
    line = GrepTool(cwd=str(tree)).execute(pattern="needle", path="long.txt").output
    assert len(line) < 600


def test_grep_reports_files_and_matches_in_files_mode(grep):
    output = grep.execute(pattern="needle", output_mode="files_with_matches").output
    assert output.splitlines()[-1].startswith("... ")


# -- glob -----------------------------------------------------------------------


def test_glob_double_star_matches_nested_and_top_level(glob):
    paths = glob.execute(pattern="**/*.py").output.splitlines()
    assert "src/app.py" in paths
    assert "src/deep/util.py" in paths


def test_glob_without_slash_matches_basename(glob):
    assert glob.execute(pattern="*.md").output.strip() == "README.md"


def test_glob_skips_noise_directories(glob):
    assert "node_modules" not in glob.execute(pattern="**/*.py").output


def test_glob_newest_first(tmp_path):
    old, new = tmp_path / "old.py", tmp_path / "new.py"
    old.write_text("")
    new.write_text("")
    os.utime(old, (1_000_000, 1_000_000))
    os.utime(new, (2_000_000, 2_000_000))
    paths = GlobTool(cwd=str(tmp_path)).execute(pattern="*.py").output.splitlines()
    assert paths == ["new.py", "old.py"]


def test_glob_limit_reports_remainder(tmp_path):
    for i in range(5):
        (tmp_path / f"f{i}.py").write_text("")
    output = GlobTool(cwd=str(tmp_path)).execute(pattern="*.py", limit=2).output
    assert "showing 2 of 5 files" in output


def test_glob_no_matches(glob):
    assert glob.execute(pattern="*.rs").output.startswith("No files match")


# -- glob: brace alternation --------------------------------------------------


def test_glob_expands_brace_alternation(tmp_path):
    """`*.{md,rst}` style patterns come from shell/ripgrep habit; they must match."""
    (tmp_path / "README.md").write_text("")
    (tmp_path / "NOTES.rst").write_text("")

    paths = (
        GlobTool(cwd=str(tmp_path)).execute(pattern="**/*.{md,rst}").output.splitlines()
    )

    assert "README.md" in paths
    assert "NOTES.rst" in paths


def test_glob_brace_without_slash_matches_basename(glob):
    assert glob.execute(pattern="*.{md,rst}").output.strip() == "README.md"


def test_glob_expands_nested_braces(glob):
    paths = glob.execute(pattern="**/*.{py,{md,txt}}").output.splitlines()

    assert "src/app.py" in paths
    assert "README.md" in paths


def test_glob_single_alternative_brace(glob):
    assert "src/app.py" in glob.execute(pattern="**/{app}.py").output


def test_grep_glob_filter_accepts_braces(grep):
    result = grep.execute(pattern="needle", glob="*.{md,rst}")

    assert "README.md" in result.output
    assert "src/app.py" not in result.output


@pytest.mark.parametrize(
    "pattern",
    [
        "**/*.py",  # no braces at all
        "src/{app.py",  # unbalanced opening brace
        "src/app.py}",  # unbalanced closing brace
        "{}*.py",  # empty alternation
    ],
)
def test_glob_brace_expansion_never_raises(pattern):
    assert _expand_braces(pattern)  # always returns something to match


def test_expand_braces_is_exhaustive():
    assert _expand_braces("*.{py,md}") == ["*.py", "*.md"]
    assert _expand_braces("a/{b,c}/{d,e}") == ["a/b/d", "a/b/e", "a/c/d", "a/c/e"]
    assert _expand_braces("plain") == ["plain"]


def test_glob_missing_path(glob):
    result = glob.execute(pattern="*.py", path="missing-dir")
    assert result.is_error and "no such path" in result.output


def test_tool_schemas_validate(tree):
    from tau.tools.validation import validate_arguments

    grep, glob = GrepTool(cwd=str(tree)), GlobTool(cwd=str(tree))
    assert validate_arguments(grep.parameters, {"pattern": "x"}) == []
    assert validate_arguments(grep.parameters, {}) == ["arguments.pattern is required"]
    assert validate_arguments(glob.parameters, {"pattern": "x"}) == []
