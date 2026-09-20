"""API end-to-end: TestClient round-trip of the small spectrum through the
full endpoint chain (upload & folder sources, all 7 steps, errors, limits)."""
from __future__ import annotations

import io
from pathlib import Path

import pytest

from conftest import write_mzxml

SMALL_ELEMENT_SPACE = {"C": (0, 35), "H": (0, 55), "N": (0, 10), "O": (0, 10)}


@pytest.fixture()
def spectrum_file(api_client, small_spectrum):
    from app.config import get_settings

    masses, ints = small_spectrum
    path = Path(get_settings().spectra_dir) / "test.mzXML"
    write_mzxml(path, masses, ints)
    return "test.mzXML"


@pytest.fixture()
def session(api_client, spectrum_file):
    resp = api_client.post("/api/sessions", json={"source": "folder", "path": spectrum_file})
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_healthz(api_client):
    resp = api_client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["models"]["cgb"] is True
    assert body["models"]["transformer"] is True


def test_index_served(api_client):
    resp = api_client.get("/")
    assert resp.status_code == 200
    assert "MEDUSA 2026" in resp.text


def test_files_and_presets_and_elements(api_client, spectrum_file):
    resp = api_client.get("/api/files")
    assert resp.status_code == 200
    paths = [f["path"] for f in resp.json()["files"]]
    assert spectrum_file in paths

    resp = api_client.get("/api/formula_presets")
    assert resp.status_code == 200
    presets = resp.json()
    assert set(presets) == {"ir_system", "pubchem10", "empty"}
    assert presets["ir_system"]["Ir"] == [1, 1]

    resp = api_client.get("/api/elements")
    assert "Ir" in resp.json()["elements"]


def test_full_chain(api_client, session, gt_formulas):
    c = api_client
    sid = session["session_id"]
    assert session["n_points"] == 100_000
    assert abs(session["mz_range"][0] - 674.36499) < 0.01

    meta = c.get(f"/api/sessions/{sid}").json()
    assert meta["n_points"] == 100_000
    assert meta["completed_steps"]["deisotope"] is False

    # --- spectrum window: decimated overview ---
    r = c.get(f"/api/sessions/{sid}/spectrum",
              params={"x0": session["mz_range"][0], "x1": session["mz_range"][1], "max_pts": 2500})
    assert r.status_code == 200
    w = r.json()
    assert len(w["masses"]) <= 2500
    assert w["decimated"] is True
    assert len(w["masses"]) == len(w["ints"])
    assert w["full_range"][0] == session["mz_range"][0]
    # strictly increasing
    assert all(b > a for a, b in zip(w["masses"], w["masses"][1:]))

    # --- exact (undecimated) small window ---
    r = c.get(f"/api/sessions/{sid}/spectrum", params={"x0": 679.36, "x1": 679.37, "max_pts": 2500})
    w2 = r.json()
    assert w2["decimated"] is False
    assert w2["n_in_window"] == len(w2["masses"])

    # --- step 2: deisotope ---
    r = c.post(f"/api/sessions/{sid}/deisotope", json={
        "algorithm": "adaptive", "z_max": 3, "min_distance": 0.01,
        "threshold": 0.2, "delta": 0.007, "n1": 2, "n2": 7,
    })
    assert r.status_code == 200, r.text
    deiso = r.json()
    assert deiso["n_ions"] >= 1
    assert len(deiso["ions"]) == deiso["n_ions"]

    # deisotoping is uncached: a re-run recomputes the same ion set
    r2 = c.post(f"/api/sessions/{sid}/deisotope", json={
        "algorithm": "adaptive", "z_max": 3, "min_distance": 0.01,
        "threshold": 0.2, "delta": 0.007, "n1": 2, "n2": 7,
    })
    assert r2.json()["n_ions"] == deiso["n_ions"]

    # ions layer present
    r = c.get(f"/api/sessions/{sid}/spectrum",
              params={"x0": 679.0, "x1": 682.5, "max_pts": 2500, "layers": "raw,ions"})
    w3 = r.json()
    assert "ion_id" in w3 and len(w3["ion_id"]) == len(w3["masses"])
    assert min(w3["ion_id"]) >= -1

    # --- step 3: elements ---
    r = c.post(f"/api/sessions/{sid}/elements", json={"element": "Ir"})
    assert r.status_code == 200, r.text
    elements = r.json()
    assert len(elements["rows"]) == deiso["n_ions"]

    # --- step 4: knee ---
    r = c.get(f"/api/sessions/{sid}/knee", params={"element": "Ir"})
    assert r.status_code == 200
    knee = r.json()
    assert 0 < knee["threshold"] <= 1
    assert 0 <= knee["knee_idx"] < knee["n_ions"]

    # --- step 5: threshold + highlight layer ---
    r = c.post(f"/api/sessions/{sid}/threshold",
               json={"element": "Ir", "source": "auto", "manual_value": None})
    assert r.status_code == 200
    thr = r.json()
    assert thr["threshold"] == knee["threshold"]
    r = c.get(f"/api/sessions/{sid}/spectrum",
              params={"x0": 679.0, "x1": 682.5, "max_pts": 2500, "layers": "raw,probs"})
    assert "prob" in r.json()

    # manual threshold
    r = c.post(f"/api/sessions/{sid}/threshold",
               json={"element": "Ir", "source": "manual", "manual_value": 1e-6})
    assert r.status_code == 200
    assert r.json()["n_points_above"] >= 0

    # --- step 6: formulas ---
    ion_id = 0
    r = c.post(f"/api/sessions/{sid}/formulas", json={
        "ion_id": ion_id, "elements": {k: list(v) for k, v in SMALL_ELEMENT_SPACE.items()},
        "mass_threshold_ppm": 4.0, "num_workers": 1,
    })
    assert r.status_code == 200, r.text
    form = r.json()
    assert form["n_candidates"] > 0
    assert form["n_valid"] > 0
    ranked = form["ranked"]
    assert len(ranked) == form["n_valid"]
    assert ranked[0]["rank"] == 1

    # ground-truth formula must be in the ranked list
    from mass_automation.formula import Formula
    gt_dict = {k: int(v) for k, v in Formula(gt_formulas[0]["formula"]).dict_formula.items() if int(v) > 0}
    found = [row for row in ranked
             if {k: int(v) for k, v in Formula(row["formula"]).dict_formula.items() if int(v) > 0} == gt_dict]
    assert found, "GT formula missing from ranked list"

    # num_workers is capped at 16
    r = c.post(f"/api/sessions/{sid}/formulas", json={
        "ion_id": ion_id, "elements": {k: list(v) for k, v in SMALL_ELEMENT_SPACE.items()},
        "mass_threshold_ppm": 4.0, "num_workers": 17,
    })
    assert r.status_code == 422

    # cached re-run
    r = c.post(f"/api/sessions/{sid}/formulas", json={
        "ion_id": ion_id, "elements": {k: list(v) for k, v in SMALL_ELEMENT_SPACE.items()},
        "mass_threshold_ppm": 4.0, "num_workers": 1,
    })
    assert r.json()["reused"] is True

    # --- step 7: compare ---
    formula_str = found[0]["formula"]
    r = c.get(f"/api/sessions/{sid}/compare", params={"ion_id": ion_id, "formula": formula_str})
    assert r.status_code == 200, r.text
    fig = r.json()
    assert "data" in fig and "layout" in fig
    assert len(fig["data"]) >= 2

    # explicit charge override is accepted (incl. negative); zero/out-of-range rejected
    r = c.get(f"/api/sessions/{sid}/compare",
              params={"ion_id": ion_id, "formula": formula_str, "charge": 1})
    assert r.status_code == 200, r.text
    r = c.get(f"/api/sessions/{sid}/compare",
              params={"ion_id": ion_id, "formula": formula_str, "charge": -1})
    assert r.status_code == 200, r.text
    zero = c.get(f"/api/sessions/{sid}/compare",
                 params={"ion_id": ion_id, "formula": formula_str, "charge": 0})
    assert zero.status_code == 400
    assert "nonzero" in zero.json()["detail"]
    assert c.get(f"/api/sessions/{sid}/compare",
                 params={"ion_id": ion_id, "formula": formula_str, "charge": -11}).status_code == 422

    # experimental trace (data[0]) uses the compare resolution: exact window
    # whenever it fits COMPARE_HARD_MAX_PTS, so far above the legacy 2500
    from app.engine.downsample import COMPARE_HARD_MAX_PTS
    exp_x = fig["data"][0]["x"]
    assert len(exp_x) > 2500
    assert len(exp_x) <= COMPARE_HARD_MAX_PTS + 10_000  # + merged peak neighborhoods
    # data[1] is the theoretical template: x = deisotopologues(isodistribution()),
    # y proportional to the theoretical relative intensities (scaled to the
    # experimental cluster base)
    from mass_automation.formula.check_formula import del_isotopologues
    t_x, t_y = del_isotopologues(*Formula(formula_str, charge=form["target_charge"]).isodistribution())
    tpl_x, tpl_y = fig["data"][1]["x"], fig["data"][1]["y"]
    assert tpl_x == [float(v) for v in t_x]
    k = max(tpl_y) / max(t_y)
    assert k > 0
    assert all(abs(a - b * k) <= 1e-6 * max(1.0, b * k) for a, b in zip(tpl_y, t_y))
    # an explicit max_pts is still honoured (legacy behaviour)
    r2 = c.get(f"/api/sessions/{sid}/compare",
               params={"ion_id": ion_id, "formula": formula_str, "max_pts": 2500})
    assert r2.status_code == 200
    assert len(r2.json()["data"][0]["x"]) <= 2500 + len(fig["data"][1]["x"]) * 200

    # session now reports completed steps
    meta = c.get(f"/api/sessions/{sid}").json()
    assert meta["completed_steps"]["deisotope"] is True
    assert meta["completed_steps"]["elements"] is True
    assert meta["completed_steps"]["formulas"] is True

    # --- delete ---
    assert c.delete(f"/api/sessions/{sid}").status_code == 200
    assert c.get(f"/api/sessions/{sid}").status_code == 404


def test_upload_flow(api_client, small_spectrum):
    c = api_client
    masses, ints = small_spectrum
    buf = io.BytesIO()
    write_mzxml(buf, masses, ints)
    data = buf.getvalue()

    r = c.post("/api/upload", files={"file": ("sample0.mzXML", data, "application/octet-stream")})
    assert r.status_code == 200, r.text
    file_id = r.json()["file_id"]

    r = c.post("/api/sessions", json={"source": "upload", "file_id": file_id})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_points"] == 100_000
    assert c.delete(f"/api/sessions/{body['session_id']}").status_code == 200


def test_upload_purge_on_session_reset(api_client, small_spectrum):
    c = api_client
    masses, ints = small_spectrum
    buf = io.BytesIO()
    write_mzxml(buf, masses, ints)
    data = buf.getvalue()

    def _upload_session(name):
        r = c.post("/api/upload", files={"file": (name, data, "application/octet-stream")})
        assert r.status_code == 200, r.text
        file_id = r.json()["file_id"]
        r = c.post("/api/sessions", json={"source": "upload", "file_id": file_id})
        assert r.status_code == 200, r.text
        return file_id, r.json()["session_id"]

    upload_dir = Path(c.app.state.settings.upload_dir)

    # default (flag off): uploaded file survives the session reset
    file_id, sid = _upload_session("purge_default.mzXML")
    assert (upload_dir / file_id).is_file()
    assert c.delete(f"/api/sessions/{sid}").status_code == 200
    assert (upload_dir / file_id).is_file(), "upload must survive by default"

    # flag on: uploaded file is removed together with the session
    from dataclasses import replace
    c.app.state.settings = replace(c.app.state.settings, delete_uploads_on_session_reset=True)
    file_id2, sid2 = _upload_session("purge_on.mzXML")
    assert c.delete(f"/api/sessions/{sid2}").status_code == 200
    assert not (upload_dir / file_id2).exists(), "upload must be purged when the flag is on"


def test_redeisotope_invalidates_downstream(api_client, spectrum_file):
    """Changing deisotoping parameters invalidates steps 3-6 results."""
    c = api_client
    r = c.post("/api/sessions", json={"source": "folder", "path": spectrum_file})
    assert r.status_code == 200, r.text
    sid = r.json()["session_id"]

    def deiso(**overrides):
        body = {"algorithm": "adaptive", "z_max": 3, "threshold": 0.2,
                "delta": 0.007, "min_distance": 0.01, "n1": 2, "n2": 7}
        body.update(overrides)
        resp = c.post(f"/api/sessions/{sid}/deisotope", json=body)
        assert resp.status_code == 200, resp.text
        return resp.json()

    FORMULAS_BODY = lambda: {
        "ion_id": 0, "elements": {k: list(v) for k, v in SMALL_ELEMENT_SPACE.items()},
        "mass_threshold_ppm": 4.0, "num_workers": 1,
    }

    deiso()
    assert c.post(f"/api/sessions/{sid}/elements", json={"element": "Ir"}).status_code == 200
    assert c.post(f"/api/sessions/{sid}/threshold",
                  json={"element": "Ir", "source": "auto", "manual_value": None}).status_code == 200
    fr = c.post(f"/api/sessions/{sid}/formulas", json=FORMULAS_BODY())
    assert fr.status_code == 200, fr.text
    assert fr.json()["reused"] is False

    # re-deisotope with different parameters: same ion ids are now different ions
    d2 = deiso(min_distance=0.05)
    assert "reused" not in d2  # step 2 is uncached

    # step 4 must refuse to run on stale probabilities
    r = c.post(f"/api/sessions/{sid}/threshold",
               json={"element": "Ir", "source": "auto", "manual_value": None})
    assert r.status_code == 400

    # step 6 must recompute (cache entry belongs to the old ion set)
    fr = c.post(f"/api/sessions/{sid}/formulas", json=FORMULAS_BODY())
    assert fr.status_code == 200, fr.text
    assert fr.json()["reused"] is False

    # step 3 recomputes, after which step 4 works again
    e = c.post(f"/api/sessions/{sid}/elements", json={"element": "Ir"})
    assert e.status_code == 200, e.text
    assert e.json()["reused"] is False
    assert c.post(f"/api/sessions/{sid}/threshold",
                  json={"element": "Ir", "source": "auto", "manual_value": None}).status_code == 200


def test_error_handling(api_client, session, spectrum_file):
    c = api_client
    sid = session["session_id"]

    assert c.get("/api/sessions/does-not-exist").status_code == 404
    r = c.post("/api/sessions", json={"source": "folder", "path": "nope.mzXML"})
    assert r.status_code == 404
    r = c.post("/api/sessions", json={"source": "folder", "path": "../../etc/passwd"})
    assert r.status_code == 400
    r = c.post("/api/sessions", json={"source": "bogus"})
    assert r.status_code == 400

    # elements before deisotoping -> 400
    r = c.post(f"/api/sessions/{sid}/elements", json={"element": "Ir"})
    assert r.status_code == 400

    # unknown element
    r = c.post(f"/api/sessions/{sid}/elements", json={"element": "Xx"})
    assert r.status_code == 400

    # bad spectrum window
    r = c.get(f"/api/sessions/{sid}/spectrum", params={"x0": 100.0, "x1": 50.0})
    assert r.status_code == 400

    # formulas: invalid element / limits
    r = c.post(f"/api/sessions/{sid}/formulas",
               json={"ion_id": 0, "elements": {"Qq": [0, 1]}})
    assert r.status_code == 400
    r = c.post(f"/api/sessions/{sid}/formulas",
               json={"ion_id": 0, "elements": {"C": [5, 1]}})
    assert r.status_code == 400
    r = c.post(f"/api/sessions/{sid}/formulas", json={"ion_id": 999, "elements": {"C": [0, 10]}})
    assert r.status_code == 400

    # manual threshold without value
    r = c.post(f"/api/sessions/{sid}/threshold",
               json={"element": "Ir", "source": "manual", "manual_value": None})
    assert r.status_code == 400


def test_unprocessable_spectrum_returns_json_error(api_client):
    """Core peak-finding chokes on degenerate (constant-m/z) spectra; the API
    must still answer with a JSON error, never a plain 500 (API contract)."""
    import numpy as np

    from app.config import get_settings

    rng = np.random.default_rng(7)
    masses = np.full(5000, 300.0)
    ints = np.abs(rng.normal(1e5, 1e4, masses.size))
    path = Path(get_settings().spectra_dir) / "noise.mzXML"
    write_mzxml(path, masses, ints)

    r = api_client.post("/api/sessions", json={"source": "folder", "path": "noise.mzXML"})
    assert r.status_code == 200, r.text
    sid = r.json()["session_id"]

    r = api_client.post(f"/api/sessions/{sid}/deisotope", json={})
    assert r.status_code == 400
    assert r.headers["content-type"].startswith("application/json")
    assert "deisotoping failed" in r.json()["detail"]


def test_session_limit(api_client, spectrum_file):
    c = api_client
    created = []
    for _ in range(4):  # MAX_ACTIVE_SESSIONS=4 in the fixture
        r = c.post("/api/sessions", json={"source": "folder", "path": spectrum_file})
        assert r.status_code == 200, r.text
        created.append(r.json()["session_id"])
    r = c.post("/api/sessions", json={"source": "folder", "path": spectrum_file})
    assert r.status_code == 409
    assert "max active sessions" in r.json()["detail"]
    # deleting one frees a slot
    assert c.delete(f"/api/sessions/{created[0]}").status_code == 200
    r = c.post("/api/sessions", json={"source": "folder", "path": spectrum_file})
    assert r.status_code == 200


def test_polarity_detection_override_and_negative_charge(api_client, small_spectrum, gt_formulas):
    from app.config import get_settings
    from mass_automation.formula import ELECTRON_MASS, Formula

    masses, ints = small_spectrum
    path = Path(get_settings().spectra_dir) / "neg.mzXML"
    write_mzxml(path, masses, ints, polarity="negative")
    r = api_client.post("/api/sessions", json={"source": "folder", "path": "neg.mzXML"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["polarity"] == "negative"
    assert body["polarity_source"] == "file"
    sid = body["session_id"]

    # manual override, validation, persistence in the session summary
    r = api_client.post(f"/api/sessions/{sid}/polarity", json={"polarity": "positive"})
    assert r.status_code == 200
    assert r.json() == {"polarity": "positive", "polarity_source": "manual"}
    assert api_client.post(f"/api/sessions/{sid}/polarity",
                           json={"polarity": "both"}).status_code == 422
    assert api_client.get(f"/api/sessions/{sid}").json()["polarity"] == "positive"

    # back to negative: deisotoping + formulas must use the signed charge
    api_client.post(f"/api/sessions/{sid}/polarity", json={"polarity": "negative"})
    r = api_client.post(f"/api/sessions/{sid}/deisotope", json={
        "algorithm": "adaptive", "z_max": 3, "min_distance": 0.01,
        "threshold": 0.2, "delta": 0.007, "n1": 2, "n2": 7,
    })
    assert r.status_code == 200, r.text
    deiso = r.json()
    r = api_client.post(f"/api/sessions/{sid}/formulas", json={
        "ion_id": 0, "elements": {k: list(v) for k, v in SMALL_ELEMENT_SPACE.items()},
        "mass_threshold_ppm": 10.0, "num_workers": 1,
    })
    assert r.status_code == 200, r.text
    form = r.json()
    tc = form["target_charge"]
    assert tc < 0
    assert abs(form["target_mass"] - (abs(tc) * form["target_mz"] + tc * ELECTRON_MASS)) < 1e-9
    gt_dict = {k: int(v) for k, v in Formula(gt_formulas[0]["formula"]).dict_formula.items() if int(v) > 0}
    found = [row for row in form["ranked"]
             if {k: int(v) for k, v in Formula(row["formula"]).dict_formula.items() if int(v) > 0} == gt_dict]
    assert found, "GT formula missing from ranked list (negative polarity)"

    # compare: default charge follows the session polarity; explicit negative works
    formula_str = found[0]["formula"]
    z_exp = f"z = {-int(deiso['ions'][0]['charge'])}"
    r = api_client.get(f"/api/sessions/{sid}/compare",
                       params={"ion_id": 0, "formula": formula_str})
    assert r.status_code == 200, r.text
    layout = r.json()["layout"]
    texts = [a.get("text", "") for a in layout.get("annotations", [])]
    texts.append((layout.get("title") or {}).get("text", ""))
    assert any(z_exp in t for t in texts)
    r = api_client.get(f"/api/sessions/{sid}/compare",
                       params={"ion_id": 0, "formula": formula_str, "charge": -1})
    assert r.status_code == 200, r.text
