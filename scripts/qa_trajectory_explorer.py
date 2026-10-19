"""Exercise the generated offline HTML in an isolated headless browser.

Optional development dependency: playwright. Uses a temporary browser profile.
"""
import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright


def check(html, browser_path, out):
    out.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path=str(browser_path), headless=True,
                                     args=['--use-angle=swiftshader', '--enable-unsafe-swiftshader'])
        context = browser.new_context(viewport={'width': 1500, 'height': 1100}, device_scale_factor=1,
                                      offline=True, accept_downloads=True)
        page = context.new_page()
        errors, network = [], []
        page.on('pageerror', lambda err: errors.append(str(err)))
        page.on('console', lambda msg: errors.append(msg.text) if msg.type == 'error' else None)
        page.on('request', lambda req: network.append(req.url) if req.url.startswith(('http:', 'https:')) else None)
        page.goto(html.resolve().as_uri())
        page.wait_for_function('state.ready', timeout=60000)

        def settle():
            page.evaluate('() => renderQueue')

        def schemas():
            issues = page.evaluate("['trajectory','distribution','distance','weights'].flatMap(id => "
                                   "(Plotly.validate($(id).data,$(id).layout)||[]).map(e=>({id,...e})))")
            assert not issues, issues

        assert page.locator('#video option').count() == 10
        assert page.locator('#emotion option').count() == 21
        schemas()
        visited = []
        for i in range(10):
            page.locator('#video').select_option(str(i))
            settle()
            assert page.evaluate('state.video') == i
            page.locator('#scrub').evaluate('(el)=>{el.value=el.max;el.dispatchEvent(new Event("input"));}')
            settle()
            assert page.evaluate('state.scene === step().vad.length-1')
            error = page.evaluate('Math.max(...currentMixture(step(),state.scene).map((x,c)=>Math.abs(x-step().prediction[c])))')
            assert error < 2e-6, error
            schemas()
            visited.append(page.evaluate('step().sample_id'))
        page.locator('#video').select_option('0')
        settle()
        page.locator('#all-paths').check()
        page.locator('#labels').check()
        page.locator('#marker-size').select_option('reference')
        page.locator('#reference').select_option('example_mean')
        page.locator('#emotion').select_option('3')
        page.locator('#distribution-mode').select_option('running')
        page.locator('#sort').select_option('distance')
        settle()
        schemas()
        assert page.evaluate("$('trajectory').data.filter(t=>t.customdata?.[0]?.kind==='other').length") == 9
        assert 'fear' in page.locator('#distance-title').inner_text()
        # A real SVG bar click should update the slider and all linked plots.
        page.locator('#weights .point').nth(2).click(force=True)
        settle()
        assert page.evaluate('state.scene') == 2
        page.locator('#near-table tr').first.click()
        settle()
        assert page.evaluate('state.emotion === nearestOrder(step(),state.scene)[0]')
        # The camera must survive scene changes and rendering updates.
        page.evaluate("Plotly.relayout('trajectory',{'scene.camera.eye':{x:2,y:-1.5,z:.8}})")
        page.locator('#next').click()
        settle()
        eye = page.evaluate("$('trajectory')._fullLayout.scene.camera.eye")
        assert abs(eye['y']+1.5) < 1e-6, eye
        page.locator('#reset-camera').click()
        page.locator('#speed').select_option('16')
        page.locator('#play').click()
        page.wait_for_function('state.scene > 3', timeout=20000)
        page.locator('#play').click()
        settle()
        assert not page.evaluate('state.playing')
        page.locator('#examples-table tr').nth(8).click()
        settle()
        assert page.evaluate('state.video') == 8
        page.locator('#all-paths').uncheck()
        page.locator('#labels').uncheck()
        page.locator('#sort').select_option('target')
        page.locator('#distribution-mode').select_option('scene')
        page.locator('#marker-size').select_option('target')
        page.locator('#reference').select_option('training_mean')
        page.locator('#scrub').evaluate('(el)=>{el.value=20;el.dispatchEvent(new Event("input"));}')
        settle()
        # Check real 3D click handler wiring for both reaction and timeline points.
        page.evaluate("$('trajectory').emit('plotly_click',{points:[{customdata:{kind:'emotion',emotion:8}}]})")
        settle()
        assert page.evaluate('state.emotion') == 8
        page.evaluate("$('trajectory').emit('plotly_click',{points:[{customdata:{kind:'scene',scene:21}}]})")
        settle()
        assert page.evaluate('state.scene') == 21
        page.evaluate('window.scrollTo(0,0)')
        page.screenshot(path=str(out/'preview_desktop.png'), full_page=True)
        page.locator('#all-paths').check()
        settle()
        page.locator('#trajectory').screenshot(path=str(out/'preview_all_ten.png'))
        page.locator('#all-paths').uncheck()
        settle()
        page.locator('details summary').click()
        with page.expect_download() as download_info:
            page.locator('#download').click()
        download_path = out/'download_check.json'
        download_info.value.save_as(download_path)
        downloaded = json.loads(download_path.read_text())
        assert len(downloaded['examples']) == 10
        download_path.unlink()
        page.locator('details summary').click()
        page.set_viewport_size({'width': 390, 'height': 844})
        page.wait_for_timeout(700)
        settle()
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        page.evaluate('window.scrollTo(0,0)')
        page.screenshot(path=str(out/'preview_mobile.png'), full_page=True)
        schemas()
        assert not errors, errors
        assert not network, network
        result = {'passed': True, 'videos_visited': visited, 'javascript_errors': errors,
                  'external_requests': network,
                  'checks': ['10 videos / 21 reactions', 'all final mixtures reconstruct predictions',
                             'Plotly schemas', 'nine background paths plus selected path',
                             'real timeline bar clicks', 'nearest-reaction and video-table clicks',
                             'camera survives scene changes', 'play / pause', '3D click handlers',
                             'reference / sorting / distribution controls', 'JSON download',
                             'desktop and mobile layouts', 'offline with no external requests']}
        (out/'browser_checks.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result))
        browser.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--html', type=Path, required=True)
    parser.add_argument('--browser', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    check(args.html, args.browser, args.out)
