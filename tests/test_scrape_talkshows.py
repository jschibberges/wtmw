from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd

import scrape_talkshows


GUIDE_HTML = """
<html>
  <body>
    <a itemprop="episode" href="/foo/folge-2"></a>
    <a itemprop="episode" href="/foo/folge-1"></a>
    <a itemprop="episode" href="/foo/folge-2"></a>
  </body>
</html>
"""

EPISODE_HTML = """
<html>
  <body>
    <ea-angaben>
      <ea-angabe-datum>So. 01.12.2024</ea-angabe-datum>
      <ea-angabe-sender>ARD</ea-angabe-sender>
    </ea-angaben>
    <h1 class="serien-titel"><a>Anne Will</a></h1>
    <h1 class="episode-title"><span itemprop="name">Klimakrise im Fokus</span></h1>
    <div class="episode-output-inhalt-inner">
      Eine ausführliche Diskussion über Klima und Wirtschaft.
    </div>
    <ul class="cast-crew">
      <li itemscope="true" itemtype="http://schema.org/Person">
        <dt itemprop="name">Dr. Alice Example (SPD)</dt>
        <dd>
          <p>Gast<br/>Bundesministerin</p>
        </dd>
      </li>
    </ul>
  </body>
</html>
"""


def test_get_episode_details_handles_empty_uid_cache(monkeypatch):
    class FakeSession:
        def get(self, url, headers=None, timeout=None):
            return SimpleNamespace(
                content=EPISODE_HTML.encode("utf-8"),
                raise_for_status=lambda: None,
            )

    monkeypatch.setattr(scrape_talkshows, "uids", set())
    monkeypatch.setattr(scrape_talkshows, "_get_thread_session", lambda: FakeSession())

    url = "https://example.com/episode-1"
    episodes = scrape_talkshows.get_episode_details([url], max_workers=1)

    assert len(episodes) == 1
    assert episodes[0]["uid"] == scrape_talkshows.create_hash(url)
    assert episodes[0]["show"] == "Anne Will"
    assert episodes[0]["station"] == "ARD"
    assert episodes[0]["guests"][0]["name"] == "Alice Example"
    assert episodes[0]["guests"][0]["party"] == "SPD"


def test_sanitize_episode_records_filters_none_and_missing_uid():
    records = [
        None,
        {"uid": "a", "title": "ok"},
        {"title": "missing"},
        "bad",
        {"uid": "b", "title": "ok2"},
    ]

    assert scrape_talkshows._sanitize_episode_records(records) == [
        {"uid": "a", "title": "ok"},
        {"uid": "b", "title": "ok2"},
    ]


def test_standardize_date_parses_german_month_names():
    assert scrape_talkshows.standardize_date("21. März 2026") == "21.03.2026"


def test_classify_guest_name_accepts_mononyms_and_aliases():
    assert scrape_talkshows.classify_guest_name("Abdelkarim") == ("accept", "mononym")
    assert scrape_talkshows.classify_guest_name("Philipp Becker alias Magic Philipp")[0] == "accept"
    assert scrape_talkshows.classify_guest_name('Jan "Monchi" Gorkow')[0] == "accept"
    assert scrape_talkshows.classify_guest_name("Kai H. Warnecke")[0] == "accept"
    assert scrape_talkshows.classify_guest_name("Rafael Laguna de la Vera")[0] == "accept"
    assert scrape_talkshows.classify_guest_name("Frédéric Prinz von Anhalt")[0] == "accept"
    assert scrape_talkshows.classify_guest_name("Detlef D! Soost")[0] == "review"


def test_classify_guest_name_rejects_clear_artifacts():
    assert scrape_talkshows.classify_guest_name(
        "Hier auf ZDFheute Nachrichten erfahrt ihr",
        "wir sorgen für Durchblick und diskutiert mit uns",
    ) == ("reject", "promo_text")
    assert scrape_talkshows.classify_guest_name("Bundesfinanzminister") == ("reject", "role_phrase")
    assert scrape_talkshows.classify_guest_name("Familie Pinzler") == ("reject", "group_label")


def test_get_episode_urls_from_guide_preserves_order_while_deduping(monkeypatch):
    class FakeSession:
        def get(self, url, headers=None, timeout=None):
            return SimpleNamespace(
                content=GUIDE_HTML.encode("utf-8"),
                raise_for_status=lambda: None,
            )

    monkeypatch.setattr(scrape_talkshows, "_get_thread_session", lambda: FakeSession())

    urls = scrape_talkshows.get_episode_urls_from_guide("https://example.com/guide")

    assert urls == [
        "https://www.fernsehserien.de/foo/folge-2",
        "https://www.fernsehserien.de/foo/folge-1",
    ]


def test_get_episode_details_skips_rejected_guest_candidates(monkeypatch):
    bad_html = """
    <html>
      <body>
        <ea-angaben>
          <ea-angabe-datum>So. 01.12.2024</ea-angabe-datum>
          <ea-angabe-sender>ZDF</ea-angabe-sender>
        </ea-angaben>
        <h1 class="serien-titel"><a>Maybrit Illner</a></h1>
        <ul class="cast-crew">
          <li itemscope="true" itemtype="http://schema.org/Person">
            <dt itemprop="name">Hier auf ZDFheute Nachrichten erfahrt ihr</dt>
            <dd>
              <p>Gast<br/>was auf der Welt passiert und diskutiert mit uns</p>
            </dd>
          </li>
        </ul>
      </body>
    </html>
    """

    class FakeSession:
        def get(self, url, headers=None, timeout=None):
            return SimpleNamespace(
                content=bad_html.encode("utf-8"),
                raise_for_status=lambda: None,
            )

    monkeypatch.setattr(scrape_talkshows, "uids", set())
    monkeypatch.setattr(scrape_talkshows, "_get_thread_session", lambda: FakeSession())

    episodes = scrape_talkshows.get_episode_details(["https://example.com/bad"], max_workers=1)

    assert len(episodes) == 1
    assert episodes[0]["guests"] == []


def test_guest_validation_override_can_force_accept_or_reject(monkeypatch, tmp_path):
    overrides_path = tmp_path / "guest_validation_overrides.json"
    overrides_path.write_text(
        json.dumps(
            {
                "Detlef D! Soost": {"status": "accept", "note": "artist-name"},
                "Familie Pinzler": {"status": "reject", "note": "group"},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(scrape_talkshows, "VALIDATION_OVERRIDES_PATH", overrides_path)
    scrape_talkshows._load_guest_validation_overrides.cache_clear()

    assert scrape_talkshows.get_guest_validation_override("Detlef D! Soost") == (
        "accept",
        "manual_override:artist-name",
    )
    assert scrape_talkshows.get_guest_validation_override("Familie Pinzler") == (
        "reject",
        "manual_override:group",
    )

    scrape_talkshows._load_guest_validation_overrides.cache_clear()


def test_export_guest_validation_review_only_outputs_unresolved_cases(monkeypatch, tmp_path):
    overrides_path = tmp_path / "guest_validation_overrides.json"
    overrides_path.write_text(
        json.dumps({"Detlef D! Soost": {"status": "accept", "note": "artist-name"}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(scrape_talkshows, "VALIDATION_OVERRIDES_PATH", overrides_path)
    scrape_talkshows._load_guest_validation_overrides.cache_clear()

    output_path = tmp_path / "guest_validation_review.xlsx"
    all_show_data = [
        {
            "show": "Show A",
            "date": "01.01.2025",
            "title": "Episode 1",
            "link": "https://example.com/1",
            "guests": [
                {"name": "Detlef D! Soost", "role": "Tänzer"},
                {"name": "Familie Pinzler", "role": "Gastfamilie"},
                {"name": "Fraktionsvorsitzender; Mitglied des Präsidiums", "role": "Politik"},
            ],
        }
    ]

    scrape_talkshows.export_guest_validation_review(all_show_data, output_path=output_path)

    review_df = pd.read_excel(output_path)
    assert review_df["cleaned_name"].tolist() == [
        "Familie Pinzler",
        "Fraktionsvorsitzender; Mitglied des Präsidiums",
    ]
    assert review_df["validation_status"].tolist() == ["reject", "review"]

    scrape_talkshows._load_guest_validation_overrides.cache_clear()


def test_scrape_episodeguide_checkpoints_incrementally(monkeypatch, tmp_path):
    saved_payloads = []

    def fake_get_episode_urls_from_guide(url):
        return ["u1", "u2", "u3"]

    def fake_get_episode_details(urls, on_episode=None, **kwargs):
        episodes = [
            {"uid": "old", "title": "existing"},
            {"uid": "new-1", "title": "one"},
            {"uid": "new-2", "title": "two"},
            {"uid": "new-3", "title": "three"},
        ][1:]
        for episode in episodes:
            if on_episode is not None:
                on_episode(episode)
        return episodes

    def fake_save_json_file(path, data):
        saved_payloads.append((path, [dict(item) for item in data]))

    monkeypatch.setattr(scrape_talkshows, "get_episode_urls_from_guide", fake_get_episode_urls_from_guide)
    monkeypatch.setattr(scrape_talkshows, "get_episode_details", fake_get_episode_details)
    monkeypatch.setattr(scrape_talkshows, "save_json_file", fake_save_json_file)

    result, stats = scrape_talkshows.scrape_fernsehserien_episodeguide(
        "https://example.com/guide",
        existing_data=[{"uid": "old", "title": "existing"}],
        output_path=tmp_path / "show.json",
        checkpoint_every=2,
    )

    assert result == [
        {"uid": "old", "title": "existing"},
        {"uid": "new-1", "title": "one"},
        {"uid": "new-2", "title": "two"},
        {"uid": "new-3", "title": "three"},
    ]
    assert len(saved_payloads) == 2
    assert saved_payloads[0][1] == [
        {"uid": "old", "title": "existing"},
        {"uid": "new-1", "title": "one"},
        {"uid": "new-2", "title": "two"},
    ]
    assert saved_payloads[1][1] == result
    assert stats["current_run_valid_episodes"] == 3
    assert stats["net_new_episodes"] == 3
