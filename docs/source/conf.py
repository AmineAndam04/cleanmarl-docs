# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'cleanmarl'
author = 'Amine Andam'

release = '0.1'
version = '0.1.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']
html_static_path = ['_static']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'


html_context = {
    "metatags": '<meta name="google-site-verification" content="pGUJ3Uw-ACan4k_IFjAEHfonB-eSEdjpkTz0yX2cc34" />'
}

def setup(app):
    app.add_html_theme('alabaster', None)  # or whatever theme you use
    app.add_config_value('google_verification', '', 'html')
    app.add_css_file('custom.css')  # optional

    app.add_html_meta({'name': 'google-site-verification',
                       'content': 'YOUR_VERIFICATION_CODE'})