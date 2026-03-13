import os
import re
import sys

if __name__ == '__main__':
    dash_path = 'templates/dashboard.html'
    port_path = 'templates/portfolio.html'

    try:
        with open(dash_path, 'r', encoding='utf-8') as f:
            dash_html = f.read()
    except (FileNotFoundError, OSError) as e:
        print(f"Error reading {dash_path}: {e}")
        sys.exit(1)


    try:
        with open(port_path, 'r', encoding='utf-8') as f:
            port_html = f.read()
    except (FileNotFoundError, OSError) as e:
        print(f"Error reading {port_path}: {e}")
        sys.exit(1)

    # 1. Replace the bypass button with auth buttons
    match = re.search(r'(<div id="authButtons">.*?</div>\s*<div id="userInfo" class="hidden flex items-center gap-3">.*?</div>)', port_html, re.DOTALL)
    if match is None:
        raise RuntimeError("Could not find authButtons/userInfo section in portfolio.html")
    auth_buttons = match.group(1)

    dash_html = re.sub(r'<!-- Viva Demo Login Bypass -->.*?</button>', auth_buttons, dash_html, flags=re.DOTALL)

    # 2. Add id="sidebarUser"
    _sidebar_pattern = (
        r'(<div)\s+(class="flex items-center justify-center lg:justify-start gap-3">'
        r'(?:(?!id=).)*?'                          # inner content up to avatar div
        r'class="w-8 h-8 rounded-full[^"]*">\s*A\s*</div>)'
    )
    _patched_html = re.sub(
        _sidebar_pattern,
        r'\1 id="sidebarUser" \2',
        dash_html,
        count=1,
        flags=re.DOTALL,
    )
    if 'id="sidebarUser"' not in _patched_html:
        raise RuntimeError(
            "Could not inject id=\"sidebarUser\": sidebar container div not found in dashboard.html"
        )
    dash_html = _patched_html

    # 3. Add Modal HTML before </body>
    match = re.search(r'(<!-- Auth Modal -->.*?)<script>', port_html, re.DOTALL)
    if match is None:
        raise RuntimeError("Could not find Auth Modal section in portfolio.html")
    auth_modal = match.group(1)
    if '<!-- Auth Modal -->' not in dash_html:
        dash_html = dash_html.replace('</body>', f'{auth_modal}\n</body>')

    # 4. Add Auth JS
    match = re.search(r'(// ==========================================\n\s*// STATE.*?checkSession\(\);)', port_html, re.DOTALL)
    if match is None:
        raise RuntimeError("Could not find Auth JS section in portfolio.html")
    auth_js = match.group(1)

    # Remove the watchlist part from the auth_js before inserting since dashboard doesn't have loadWatchlist()
    auth_js = re.sub(r'loadWatchlist\(\);', '', auth_js)

    if 'currentAuthMode' not in dash_html:
        dash_html = re.sub(r'(</script>)\s*(</body>)', f'{auth_js}\n    \\1\n\\2', dash_html, count=1)

    # 5. Atomic write: write to a temp file in the same directory, then os.replace
    _dir = os.path.dirname(os.path.abspath(dash_path))
    _tmp_path = os.path.join(_dir, f'.{os.path.basename(dash_path)}.tmp')
    try:
        with open(_tmp_path, 'w', encoding='utf-8') as f:
            f.write(dash_html)
        os.replace(_tmp_path, dash_path)
    except OSError as e:
        # Clean up temp file if it survived
        if os.path.exists(_tmp_path):
            os.remove(_tmp_path)
        print(f"Error writing {dash_path}: {e}")
        sys.exit(1)

    # 6. Readback verification
    with open(dash_path, 'r', encoding='utf-8') as f:
        _written = f.read()
    if _written != dash_html:
        raise RuntimeError(f"Readback mismatch: {dash_path} does not match the patched content.")

    print("Dashboard patched successfully.")

