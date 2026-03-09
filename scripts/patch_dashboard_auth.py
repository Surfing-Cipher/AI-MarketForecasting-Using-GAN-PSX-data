import re

dash_path = 'templates/dashboard.html'
port_path = 'templates/portfolio.html'

with open(dash_path, 'r') as f:
    dash_html = f.read()

with open(port_path, 'r') as f:
    port_html = f.read()

# 1. Replace the bypass button with auth buttons
auth_buttons = re.search(r'(<div id="authButtons">.*?</div>\s*<div id="userInfo" class="hidden flex items-center gap-3">.*?</div>)', port_html, re.DOTALL).group(1)

dash_html = re.sub(r'<!-- Viva Demo Login Bypass -->.*?</button>', auth_buttons, dash_html, flags=re.DOTALL)

# 2. Add id="sidebarUser"
dash_html = dash_html.replace(
    '<div class="flex items-center justify-center lg:justify-start gap-3">\n                <div\n                    class="w-8 h-8 rounded-full bg-gradient-to-tr from-neon-cyan to-neon-purple flex items-center justify-center font-bold text-xs text-black">\n                    A</div>',
    '<div id="sidebarUser" class="flex items-center justify-center lg:justify-start gap-3">\n                <div\n                    class="w-8 h-8 rounded-full bg-gradient-to-tr from-neon-cyan to-neon-purple flex items-center justify-center font-bold text-xs text-black">\n                    A</div>'
)

# 3. Add Modal HTML before </body>
auth_modal = re.search(r'(<!-- Auth Modal -->.*?)<script>', port_html, re.DOTALL).group(1)
if '<!-- Auth Modal -->' not in dash_html:
    dash_html = dash_html.replace('</body>', f'{auth_modal}\n</body>')

# 4. Add Auth JS
auth_js = re.search(r'(// ==========================================\n\s*// STATE.*?checkSession\(\);)', port_html, re.DOTALL).group(1)

# Remove the watchlist part from the auth_js before inserting since dashboard doesn't have loadWatchlist()
auth_js = re.sub(r'loadWatchlist\(\);', '', auth_js)

if 'currentAuthMode' not in dash_html:
    dash_html = dash_html.replace('</script>\n</body>', f'{auth_js}\n    </script>\n</body>')

with open(dash_path, 'w') as f:
    f.write(dash_html)

print("Dashboard patched successfully.")
