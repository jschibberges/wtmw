import reflex as rx
from reflex.plugins.sitemap import SitemapPlugin


config = rx.Config(
    app_name="wtmw_reflex",
    state_auto_setters=False,
    disable_plugins=[SitemapPlugin],
)
