import os

index_file = 'templates/index.html'

with open(index_file, 'r', encoding='utf-8') as f:
    text = f.read()

# Find the three sections
live_start = text.find('<div id="view-live"')
dashboard_start = text.find('<div id="view-dashboard"')
settings_start = text.find('<div id="view-settings"')
script_start = text.find('<script>')

live_html = text[live_start:dashboard_start].replace('class="view-section active layout"', 'class="layout"')
dash_html = text[dashboard_start:settings_start].replace('class="view-section dashboard-layout"', 'class="dashboard-layout" style="display:block;"')
set_html = text[settings_start:script_start].replace('class="view-section settings-layout"', 'class="settings-layout" style="display:block;"')

# Create base.html
base_top = text[:live_start]
base_bottom = text[script_start:]

# Modify nav-tabs in base_top to use real hrefs
base_top = base_top.replace('onclick="switchTab(\'live\')"', 'href="/live"').replace('onclick="switchTab(\'dashboard\')"', 'href="/"').replace('onclick="switchTab(\'settings\')"', 'href="/settings"')
base_top = base_top.replace('<div class="nav-tab active"', '<a class="nav-tab {% if active_page == \'live\' %}active{% endif %}"')
base_top = base_top.replace('<div class="nav-tab"', '<a class="nav-tab {% if active_page == \'dashboard\' %}active{% endif %}"', 1)
base_top = base_top.replace('<div class="nav-tab"', '<a class="nav-tab {% if active_page == \'settings\' %}active{% endif %}"', 1)
base_top = base_top.replace('</div>\n      </nav>', '</a>\n      </nav>')
# We need to fix closing divs for nav tabs
base_top = base_top.replace('</a>\n        <a class="nav-tab', '</a>\n        <a class="nav-tab') # manual fixing might be needed

import re
base_top = re.sub(r'<div class="nav-tab(.*?)" onclick="switchTab\(\'(.*?)\'\)">(.*?)</div>',
                  r'<a href="/\2" class="nav-tab {% if active_page == \'\2\' %}active{% endif %}" style="text-decoration:none;">\3</a>', 
                  text[:live_start])
# replace actual hrefs: /live-> /live, /dashboard -> /, /settings -> /settings
base_top = base_top.replace('href="/dashboard"', 'href="/"')

base_html = base_top + '\n  {% block content %}{% endblock %}\n\n' + base_bottom

# Remove switchTab from JS
base_html = re.sub(r'function switchTab\(.*?\n    }', '', base_html, flags=re.DOTALL)

with open('templates/base.html', 'w', encoding='utf-8') as f:
    f.write(base_html)

with open('templates/dashboard.html', 'w', encoding='utf-8') as f:
    f.write("{% extends 'base.html' %}\n{% set active_page = 'dashboard' %}\n{% block content %}\n" + dash_html + "\n<script>window.onload = loadDashboard;</script>\n{% endblock %}")

with open('templates/live.html', 'w', encoding='utf-8') as f:
    f.write("{% extends 'base.html' %}\n{% set active_page = 'live' %}\n{% block content %}\n" + live_html + "\n{% endblock %}")

with open('templates/settings.html', 'w', encoding='utf-8') as f:
    f.write("{% extends 'base.html' %}\n{% set active_page = 'settings' %}\n{% block content %}\n" + set_html + "\n<script>window.onload = loadSettings;</script>\n{% endblock %}")

# Update app.py
with open('app.py', 'r', encoding='utf-8') as f:
    app_text = f.read()

new_routes = """
@app.route("/")
def index():
    return render_template("dashboard.html")

@app.route("/live")
def route_live():
    return render_template("live.html")

@app.route("/settings")
def route_settings():
    return render_template("settings.html")
"""
app_text = re.sub(r'@app\.route\("/"\)\ndef index\(\):\n    return render_template\("index\.html"\)', new_routes.strip(), app_text)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(app_text)

print("Migration completed.")
