import os
from dnnkit.paper import generate_report

def test_generate_report_no_runs(tmp_path):
    out_file = tmp_path / "report.md"
    generate_report(outputs_dir=str(tmp_path), output_file=str(out_file))
    assert os.path.exists(out_file)
