"""Check that interrupted analysis output cannot replace completed results."""
import pytest

from file_paths import atomic_output_path


def test_atomic_output_preserves_completed_result_if_writer_fails(tmp_path):
    output = tmp_path / 'labels.csv'
    output.write_text('completed data', encoding='utf-8')
    with pytest.raises(OSError, match='disk write failed'), atomic_output_path(output) as temporary:
        temporary.write_text('partially written data', encoding='utf-8')
        raise OSError('disk write failed')
    assert output.read_text(encoding='utf-8') == 'completed data'
    assert list(tmp_path.iterdir()) == [output]


def test_atomic_output_publishes_complete_result(tmp_path):
    output = tmp_path / 'analysis' / 'metrics.json'
    with atomic_output_path(output) as temporary:
        temporary.write_text('{}', encoding='utf-8')
        assert not output.exists()
    assert output.read_text(encoding='utf-8') == '{}'
    assert list(output.parent.iterdir()) == [output]
