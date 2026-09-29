import zipfile

from pfas_risk import fetch


def test_record_source_and_change_detection(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "SOURCE_RECORD", tmp_path / "massdep_source.json")
    z = tmp_path / "release.zip"
    with zipfile.ZipFile(z, "w") as f:
        info = zipfile.ZipInfo("RELEASE.DBF", date_time=(2026, 5, 29, 17, 20, 0))
        f.writestr(info, b"data")
    assert fetch.recorded_sha256() is None           # nothing built yet: always "changed"
    fetch.record_source(z, n_releases=3)
    assert fetch.recorded_sha256() == fetch.sha256(z)
    record = fetch.SOURCE_RECORD.read_text()
    assert '"tables_dated": "2026-05-29"' in record and '"pfas_releases": 3' in record
