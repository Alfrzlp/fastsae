#!/usr/bin/env python3
"""Generate combined index.html with sidebar navigation."""

content = open('benchmark.html').read()

# The benchmark.html already has sidebar, just update title
content = content.replace(
    '<title>fastsae — Benchmark Comparison</title>',
    '<title>fastsae — Fast Small Area Estimation</title>'
)

# Update nav title
content = content.replace('>Benchmark<', '>Performance<')

# Change logo link to just #
content = content.replace(
    'href="index.html" class="sidebar-logo"',
    'href="#overview" class="sidebar-logo"'
)

# Add CSS for content sections in the style block
section_css = """
    .content-section { margin-bottom: 48px; }
    .content-section h2 { font-size: 20px; font-weight: 600; color: var(--fg); margin: 32px 0 16px; padding-top: 32px; border-top: 1px solid var(--border); }
    .content-section h2:first-child { border-top: none; margin-top: 0; padding-top: 0; }
    .content-header { margin-bottom: 32px; }
    .content-header h1 { font-size: 32px; font-weight: 700; margin-bottom: 12px; }
    .content-header .subtitle { color: var(--fg-secondary); margin-bottom: 20px; }
    .kw { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 16px; }
    .k { background: var(--bg-secondary); border: 1px solid var(--border); font-size: 13px; font-weight: 500; padding: 4px 12px; border-radius: 20px; color: var(--fg-secondary); }
    pre { background: var(--bg-secondary); border: 1px solid var(--border); border-radius: 8px; padding: 16px; overflow-x: auto; margin: 12px 0; }
    pre code { font: 14px/1.5 monospace; color: var(--fg); background: none; border: none; padding: 0; }
    table { width: 100%; border-collapse: collapse; font-size: 14px; margin: 12px 0; }
    th { text-align: left; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: var(--fg-muted); padding: 8px 12px; border-bottom: 1px solid var(--border); }
    td { padding: 10px 12px; border-bottom: 1px solid var(--border); color: var(--fg-secondary); }
    td:first-child { color: var(--fg); }
    code { font: 14px/1.5 monospace; background: var(--bg-secondary); padding: 2px 6px; border-radius: 4px; border: 1px solid var(--border); }
    p { color: var(--fg-secondary); margin-bottom: 12px; }
    .refs-section { margin-top: 48px; padding-top: 32px; border-top: 1px solid var(--border); }
    .refs-section ul { margin-left: 20px; color: var(--fg-secondary); }
    .refs-section li { margin-bottom: 8px; }
"""

# Insert CSS before .chart-card
content = content.replace('.chart-card {', section_css + '.chart-card {')

# Build doc sections
doc_sections = '''
    <section id="overview" class="content-section">
      <div class="content-header">
        <span class="badge">fastsae · sae · emdi</span>
        <h1>Fast Small Area Estimation</h1>
        <p class="subtitle">High-performance R package for Small Area Estimation using Fay-Herriot, Spatial Fay-Herriot, and Battese-Harter-Fuller models powered by C++.</p>
        <div class="kw">
          <span class="k">up to 12,000x faster</span>
          <span class="k">RcppArmadillo</span>
          <span class="k">Parallel bootstrap</span>
          <span class="k">MIT License</span>
        </div>
      </div>
    </section>

    <section id="install" class="content-section">
      <h2>Installation</h2>
      <pre><code>install.packages("remotes")
remotes::install_github("ridsonap/fastsae")</code></pre>
    </section>

    <section id="models" class="content-section">
      <h2>Supported Models</h2>
      <table>
        <tr><th>Function</th><th>Model</th><th>MSE</th></tr>
        <tr><td><code>eblup_area()</code></td><td>Fay-Herriot (area-level)</td><td>Analytical</td></tr>
        <tr><td><code>seblup_area()</code></td><td>Spatial Fay-Herriot</td><td>Analytical, PBMSE, NPBMSE</td></tr>
        <tr><td><code>eblup_unit()</code></td><td>Battese-Harter-Fuller (unit-level)</td><td>PBMSE</td></tr>
      </table>
    </section>

    <section id="eblup_area" class="content-section">
      <h2>eblup_area - Fay-Herriot</h2>
      <pre><code>library(fastsae)
m <- eblup_area(
  formula = y ~ x1 + x2 + x3,
  data    = survey_data,
  vardir  = "vardir",
  method  = "REML"
)</code></pre>
      <p>Output: <code>df_eblup</code>, <code>estcoef</code>, <code>random_effect_var</code>, <code>goodness</code>, <code>n_iter</code>, <code>convergence</code></p>
    </section>

    <section id="seblup_area" class="content-section">
      <h2>seblup_area - Spatial Fay-Herriot</h2>
      <pre><code>library(fastsae)
m <- seblup_area(
  formula = y ~ x1 + x2 + x3,
  data    = survey_data,
  vardir  = "vardir",
  W       = W_matrix,
  method  = "REML"
)

# With Parametric Bootstrap MSE
m_pb <- seblup_area(
  formula    = y ~ x1 + x2 + x3,
  data       = survey_data,
  vardir     = "vardir",
  W          = W_matrix,
  mse_method = "pbmse",
  B          = 200,
  n_threads  = 4,
  seed       = 42
)</code></pre>
    </section>

    <section id="eblup_unit" class="content-section">
      <h2>eblup_unit - Battese-Harter-Fuller</h2>
      <pre><code>library(fastsae)
m <- eblup_unit(
  formula     = CornHec ~ CornPix + SoyBeansPix,
  unit_data   = survey,
  Xpop        = population_means,
  domain_var  = "CountyIndex",
  popsize_var = "PopnSegments"
)</code></pre>
    </section>

    <section id="refs" class="content-section refs-section">
      <h2>References</h2>
      <ul>
        <li>Fay, R. E., & Herriot, R. A. (1979). Estimates of income for small places. <em>JASA</em>, 74(366), 269-277.</li>
        <li>Battese, G. E., Harter, R. M., & Fuller, W. A. (1988). An error-components model. <em>JASA</em>, 83(401), 28-36.</li>
        <li>Rao, J. N. K., & Molina, I. (2015). <em>Small Area Estimation</em>, 2nd Ed. Wiley.</li>
      </ul>
    </section>

    <section id="benchmark" class="content-section">
'''

# Add documentation sections before benchmark section
content = content.replace('<div class="chart-card">', doc_sections + '<div class="chart-card">')

# Close section tag after chart-card div
content = content.replace(
    '</div>\n\n  </div>\n\n  <button class="theme-toggle">',
    '</div>\n    </section>\n  </div>\n\n  <button class="theme-toggle">'
)

with open('index.html', 'w') as f:
    f.write(content)

print('Generated docs/index.html successfully!')
