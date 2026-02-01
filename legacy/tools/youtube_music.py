"""
YouTube Music integration for voice assistant.
Requires ytmusicapi and authentication setup.
"""
import os
import json
import shutil
import sys
from pathlib import Path
from http.cookies import SimpleCookie
from ytmusicapi import YTMusic
from tools.audio import pause_media, clear_media_pause_state, is_media_paused_by_assistant

_active_headless_process = None
_AUTO_AUTH_HEADERS = None


def _stop_headless_playback():
    global _active_headless_process
    if not _active_headless_process:
        return

    try:
        if _active_headless_process.poll() is None:
            _active_headless_process.terminate()
            try:
                _active_headless_process.wait(timeout=1.5)
            except Exception:
                _active_headless_process.kill()
    finally:
        _active_headless_process = None

def _resolve_auth_file():
    auth_path = os.getenv("YTMUSIC_AUTH_FILE")
    if auth_path:
        candidate = Path(auth_path).expanduser()
        if candidate.is_file():
            return candidate

    cwd_candidate = Path.cwd() / "oauth.json"
    if cwd_candidate.is_file():
        return cwd_candidate

    for parent in Path(__file__).resolve().parents:
        candidate = parent / "oauth.json"
        if candidate.is_file():
            return candidate

    return None

def _load_auth_headers(auth_path):
    if not auth_path:
        return None
    try:
        with open(auth_path, "r", encoding="utf-8") as auth_file:
            data = json.load(auth_file)
    except Exception:
        return None
    if isinstance(data, dict) and isinstance(data.get("headers"), dict):
        return data["headers"]
    if isinstance(data, dict):
        return data
    return None

def _get_header(headers, name):
    if not headers:
        return None
    target = name.lower()
    for key, value in headers.items():
        if key.lower() == target:
            return value
    return None

def _resolve_cookie_source(auth_headers):
    cookies_from_browser = os.getenv("YTMUSIC_COOKIES_FROM_BROWSER")
    if cookies_from_browser:
        return "browser", cookies_from_browser

    cookies_path = os.getenv("YTMUSIC_COOKIES_PATH")
    if cookies_path:
        candidate = Path(cookies_path).expanduser()
        if candidate.is_file():
            return "file", str(candidate)

    cookie_header = _get_header(auth_headers, "cookie")
    if cookie_header:
        return "header", cookie_header

    return None, None

def _build_stream_headers(auth_headers):
    if not auth_headers:
        return {}
    headers = {}
    cookie = _get_header(auth_headers, "cookie")
    if cookie:
        headers["Cookie"] = cookie
    user_agent = _get_header(auth_headers, "user-agent")
    if user_agent:
        headers["User-Agent"] = user_agent
    origin = _get_header(auth_headers, "origin") or _get_header(auth_headers, "x-origin")
    if origin:
        headers["Origin"] = origin
    return headers

def _build_ytdlp_auth_args(auth_headers):
    cookie_source, cookie_value = _resolve_cookie_source(auth_headers)
    if not cookie_source:
        return []
    args = []
    if cookie_source == "browser":
        args += ["--cookies-from-browser", cookie_value]
    elif cookie_source == "file":
        args += ["--cookies", cookie_value]
    elif cookie_source == "header":
        for key, value in _build_stream_headers(auth_headers).items():
            if key.lower() == "user-agent":
                args += ["--user-agent", value]
            else:
                args += ["--add-header", f"{key}:{value}"]
    return args

def _build_mpv_raw_options(auth_headers):
    options = []
    cookie_source, cookie_value = _resolve_cookie_source(auth_headers)
    if cookie_source == "browser":
        options.append(f"cookies-from-browser={cookie_value}")
    elif cookie_source == "file":
        options.append(f"cookies={cookie_value}")
    elif cookie_source == "header":
        for key, value in _build_stream_headers(auth_headers).items():
            if key.lower() == "user-agent":
                options.append(f"user-agent={value}")
            else:
                options.append(f"add-header={key}:{value}")

    js_runtime = _env_value("YTMUSIC_YTDLP_JS_RUNTIME")
    if js_runtime:
        options.append(f"js-runtimes={js_runtime}")
    remote_components = _env_value("YTMUSIC_YTDLP_REMOTE_COMPONENTS")
    if remote_components:
        options.append(f"remote-components={remote_components}")
    extractor_args = _build_ytdlp_extractor_arg_string()
    if extractor_args:
        options.append(f"extractor-args={extractor_args}")

    return options

def _env_value(name):
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value or None

def _resolve_ytdlp_path():
    explicit_path = _env_value("YTMUSIC_YTDLP_PATH")
    if explicit_path:
        return explicit_path
    try:
        candidate = Path(sys.executable).resolve().parent / "yt-dlp"
        if candidate.is_file():
            return str(candidate)
    except Exception:
        pass
    return shutil.which("yt-dlp")

def _is_truthy_env(name):
    value = _env_value(name)
    if not value:
        return False
    return value.lower() not in ("0", "false", "no")

def _build_ytdlp_extractor_args():
    args = []
    player_client = _env_value("YTMUSIC_YTDLP_PLAYER_CLIENT")
    if player_client:
        args.append(f"youtube:player_client={player_client}")

    pot_suffix = ";disable_innertube=1" if _is_truthy_env("YTMUSIC_YTDLP_POT_DISABLE_INNERTUBE") else ""
    pot_script_path = _env_value("YTMUSIC_YTDLP_POT_SCRIPT_PATH")
    if pot_script_path:
        args.append(f"youtubepot-bgutilscript:script_path={pot_script_path}{pot_suffix}")

    pot_http = _env_value("YTMUSIC_YTDLP_POT_HTTP")
    if pot_http:
        args.append(f"youtubepot-bgutilhttp:base_url={pot_http}{pot_suffix}")

    return args

def _build_ytdlp_runtime_args():
    args = []
    js_runtime = _env_value("YTMUSIC_YTDLP_JS_RUNTIME")
    if js_runtime:
        args += ["--js-runtimes", js_runtime]
    remote_components = _env_value("YTMUSIC_YTDLP_REMOTE_COMPONENTS")
    if remote_components:
        args += ["--remote-components", remote_components]
    for extractor_arg in _build_ytdlp_extractor_args():
        args += ["--extractor-args", extractor_arg]
    return args

def _build_ytdlp_extractor_arg_string():
    args = _build_ytdlp_extractor_args()
    return ";".join(args) if args else None

def _build_mpv_audio_args():
    args = []
    audio_out = _env_value("YTMUSIC_MPV_AO")
    if audio_out:
        args.append(f"--ao={audio_out}")
    audio_device = _env_value("YTMUSIC_MPV_AUDIO_DEVICE")
    if audio_device:
        args.append(f"--audio-device={audio_device}")
    volume = _env_value("YTMUSIC_MPV_VOLUME")
    if volume:
        args.append(f"--volume={volume}")
    log_path = _env_value("YTMUSIC_MPV_LOG_PATH")
    if log_path:
        args.append(f"--log-file={log_path}")
    return args

def _cookie_header_has_sapisid(cookie_header):
    if not cookie_header:
        return False
    cookie = SimpleCookie()
    try:
        cookie.load(cookie_header.replace('"', ""))
    except Exception:
        return False
    return "__Secure-3PAPISID" in cookie

def _cookie_header_from_jar(jar):
    if not jar:
        return None
    cookies_by_name = {}
    for cookie in jar:
        domain = cookie.domain.lstrip(".")
        if domain.endswith("youtube.com"):
            cookies_by_name.setdefault(cookie.name, cookie.value)
    if "__Secure-3PAPISID" not in cookies_by_name:
        for cookie in jar:
            if cookie.name == "__Secure-3PAPISID":
                cookies_by_name[cookie.name] = cookie.value
                break
    if not cookies_by_name:
        return None
    return "; ".join(f"{name}={value}" for name, value in cookies_by_name.items())

def _default_browser_candidates():
    import platform
    system = platform.system()
    if system == "Darwin":
        return ["chrome", "brave", "edge", "chromium", "vivaldi", "opera", "safari", "firefox"]
    if system == "Windows":
        return ["chrome", "edge", "brave", "chromium", "vivaldi", "opera", "firefox"]
    return ["chrome", "chromium", "brave", "edge", "firefox", "opera", "vivaldi"]

def _parse_browser_spec(browser_spec):
    if not browser_spec:
        return None, None
    if ":" in browser_spec:
        name, profile = browser_spec.split(":", 1)
        return name.strip(), profile.strip() or None
    return browser_spec.strip(), None

def _build_auto_auth_headers(cookie_header):
    from ytmusicapi.helpers import initialize_headers
    headers = {
        "cookie": cookie_header,
        "x-goog-authuser": os.getenv("YTMUSIC_GOOG_AUTHUSER", "0")
    }
    headers.update(initialize_headers())
    return headers

def _auto_auth_headers_from_browser():
    global _AUTO_AUTH_HEADERS
    if _AUTO_AUTH_HEADERS is not None:
        return _AUTO_AUTH_HEADERS

    auto_enabled = os.getenv("YTMUSIC_AUTO_BROWSER_AUTH", "true").lower() not in ("0", "false", "no")
    if not auto_enabled:
        _AUTO_AUTH_HEADERS = None
        return None

    try:
        from yt_dlp import cookies as ytdlp_cookies
    except Exception:
        _AUTO_AUTH_HEADERS = None
        return None

    browser_spec = os.getenv("YTMUSIC_COOKIES_FROM_BROWSER") or os.getenv("YTMUSIC_BROWSER")
    browser_name, profile_from_spec = _parse_browser_spec(browser_spec)
    browser_profile = os.getenv("YTMUSIC_BROWSER_PROFILE") or profile_from_spec
    browser_keyring = os.getenv("YTMUSIC_BROWSER_KEYRING")
    browser_container = os.getenv("YTMUSIC_BROWSER_CONTAINER")

    candidates = [browser_name] if browser_name else _default_browser_candidates()
    for candidate in candidates:
        if not candidate:
            continue
        try:
            jar = ytdlp_cookies.extract_cookies_from_browser(
                candidate,
                profile=browser_profile,
                keyring=browser_keyring,
                container=browser_container
            )
        except Exception:
            continue
        cookie_header = _cookie_header_from_jar(jar)
        if not _cookie_header_has_sapisid(cookie_header):
            continue
        _AUTO_AUTH_HEADERS = _build_auto_auth_headers(cookie_header)
        return _AUTO_AUTH_HEADERS

    _AUTO_AUTH_HEADERS = None
    return None

def _has_min_auth_headers(headers):
    return bool(headers and _get_header(headers, "cookie") and _get_header(headers, "x-goog-authuser"))

class YouTubeMusicManager:
    def __init__(self):
        auth_path = _resolve_auth_file()
        file_headers = _load_auth_headers(auth_path)
        auto_headers = _auto_auth_headers_from_browser()
        self.auth_headers = auto_headers or file_headers
        if self.auth_headers and _has_min_auth_headers(self.auth_headers):
            try:
                self.ytmusic = YTMusic(auth=self.auth_headers)
            except Exception:
                self.ytmusic = YTMusic()
                self.auth_headers = None
        else:
            self.ytmusic = YTMusic()
    
    def search_and_play(self, query: str, search_type: str = "songs"):
        """
        Search for content and return playable URL.
        
        Args:
            query: Search query (song name, artist, etc.)
            search_type: Type of content ("songs", "videos", "albums", "artists")
        
        Returns:
            Dictionary with video ID and URL
        """
        
        try:
            results = self.ytmusic.search(query, filter=search_type, limit=1)
            if not results:
                return {"error": f"No results found for '{query}'"}
            
            result = results[0]
            video_id = result.get('videoId')
            
            if not video_id:
                return {"error": "Could not get video ID from search result"}
            
            return {
                "video_id": video_id,
                "url": f"https://music.youtube.com/watch?v={video_id}",
                "title": result.get('title', 'Unknown'),
                "artist": result.get('artists', [{}])[0].get('name', 'Unknown') if result.get('artists') else 'Unknown'
            }
        except Exception as e:
            return {"error": f"Search failed: {str(e)}"}
    
    def play_playlist(self, playlist_name: str):
        """
        Find and return playlist URL.
        
        Args:
            playlist_name: Name of the playlist to search for
        
        Returns:
            Dictionary with playlist ID and URL
        """
        
        try:
            results = self.ytmusic.search(playlist_name, filter="playlists", limit=1)
            if not results:
                return {"error": f"No playlist found for '{playlist_name}'"}
            
            result = results[0]
            playlist_id = result.get('browseId')
            
            if not playlist_id:
                return {"error": "Could not get playlist ID from search result"}
            
            return {
                "playlist_id": playlist_id,
                "url": f"https://music.youtube.com/playlist?list={playlist_id}",
                "title": result.get('title', 'Unknown')
            }
        except Exception as e:
            return {"error": f"Playlist search failed: {str(e)}"}

def play_youtube_music(query: str, content_type: str = "song"):
    """
    Play music on YouTube Music.
    
    Args:
        query: What to play (song name, artist, playlist name, etc.)
        content_type: Type of content - "song", "video", "album", "artist", or "playlist"
    
    Returns:
        Result message with playback information
    """
    global _active_headless_process
    import subprocess
    import platform
    
    manager = YouTubeMusicManager()
    headless_mode = os.getenv("HEADLESS_PLAYBACK", "false").lower() == "true"

    # Stop any assistant-paused media from resuming when we start new playback.
    if not is_media_paused_by_assistant():
        pause_media()
    clear_media_pause_state()
    
    type_mapping = {
        "song": "songs",
        "video": "videos",
        "album": "albums",
        "artist": "artists",
        "playlist": "playlists"
    }
    
    if content_type == "playlist":
        result = manager.play_playlist(query)
    else:
        search_type = type_mapping.get(content_type, "songs")
        result = manager.search_and_play(query, search_type)
    
    if "error" in result:
        return result["error"]
    
    url = result.get("url")
    if not url:
        return "Failed to get playback URL"
    
    try:
        if headless_mode:
            _stop_headless_playback()
            # Headless mode: stream audio directly using mpv or yt-dlp + ffplay.
            mpv_path = shutil.which("mpv")
            ytdlp_path = _resolve_ytdlp_path()
            ffplay_path = shutil.which("ffplay")

            if mpv_path:
                mpv_args = [mpv_path, "--no-video", "--really-quiet"]
                if ytdlp_path:
                    mpv_args.append(f"--script-opts=ytdl_hook-ytdl_path={ytdlp_path}")
                mpv_args += _build_mpv_audio_args()
                raw_options = _build_mpv_raw_options(manager.auth_headers)
                if raw_options:
                    mpv_args.append(f"--ytdl-raw-options={','.join(raw_options)}")
                mpv_args.append(url)
                _active_headless_process = subprocess.Popen(
                    mpv_args,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                if content_type == "playlist":
                    return f"Playing playlist: {result.get('title', query)}"
                return f"Now playing: {result.get('title', 'Unknown')} by {result.get('artist', 'Unknown')}"

            if not ytdlp_path:
                return "Headless playback requires mpv or yt-dlp+ffplay. Install with: sudo apt-get install mpv"

            if not ffplay_path:
                return "Headless playback requires ffplay (ffmpeg). Install with: sudo apt-get install ffmpeg"

            # Fallback to yt-dlp + ffplay
            try:
                ytdlp_args = [ytdlp_path, "-f", "bestaudio", "-g"]
                if content_type == "playlist":
                    ytdlp_args += ["--playlist-items", "1"]
                ytdlp_args += _build_ytdlp_auth_args(manager.auth_headers)
                ytdlp_args += _build_ytdlp_runtime_args()
                ytdlp_args.append(url)

                # Use yt-dlp to get direct audio stream URL
                audio_output = subprocess.check_output(
                    ytdlp_args,
                    stderr=subprocess.DEVNULL,
                    text=True
                )
                audio_url = ""
                for line in audio_output.splitlines():
                    if line.strip():
                        audio_url = line.strip()
                        break
                if not audio_url:
                    return "Failed to get audio stream URL for headless playback."

                # Play with ffplay (part of ffmpeg)
                _active_headless_process = subprocess.Popen(
                    [ffplay_path, "-nodisp", "-autoexit", "-loglevel", "quiet", audio_url],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                if content_type == "playlist":
                    return f"Playing first track from playlist: {result.get('title', query)} (install mpv for full playlist)"
                return f"Now playing: {result.get('title', 'Unknown')} by {result.get('artist', 'Unknown')}"
            except subprocess.CalledProcessError:
                return "Headless playback failed to resolve stream URL. Try installing mpv."
        else:
            # Browser mode: open in default browser
            system = platform.system()
            if system == "Darwin":
                subprocess.Popen(["open", url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            elif system == "Linux":
                subprocess.Popen(["xdg-open", url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                return f"Unsupported platform. Open this URL manually: {url}"
            
            if content_type == "playlist":
                return f"Opening playlist: {result.get('title', query)}"
            else:
                return f"Now playing: {result.get('title', 'Unknown')} by {result.get('artist', 'Unknown')}"
    except Exception as e:
        return f"Playback failed: {str(e)}. URL: {url}"

def stop_music():
    """
    Stop all audio playback including music and text-to-speech.
    Kills mpv, ffplay, pyttsx3, espeak, and other audio processes.
    
    Returns:
        Result message indicating what was stopped
    """
    import subprocess
    import platform
    
    stopped = []
    
    try:
        _stop_headless_playback()
        clear_media_pause_state()

        # Kill mpv (headless mode music)
        result = subprocess.run(
            ["pkill", "-9", "mpv"],
            capture_output=True
        )
        if result.returncode == 0:
            stopped.append("mpv")
        
        # Kill ffplay (alternative headless mode)
        result = subprocess.run(
            ["pkill", "-9", "ffplay"],
            capture_output=True
        )
        if result.returncode == 0:
            stopped.append("ffplay")
        
        # Kill espeak (TTS backend on Linux)
        result = subprocess.run(
            ["pkill", "-9", "espeak"],
            capture_output=True
        )
        if result.returncode == 0:
            stopped.append("espeak")
        
        # Kill espeak-ng (newer TTS backend on Linux)
        result = subprocess.run(
            ["pkill", "-9", "espeak-ng"],
            capture_output=True
        )
        if result.returncode == 0:
            stopped.append("espeak-ng")
        
        # Kill festival (alternative TTS on Linux)
        result = subprocess.run(
            ["pkill", "-9", "festival"],
            capture_output=True
        )
        if result.returncode == 0:
            stopped.append("festival")
        
        # Kill flite (lightweight TTS)
        result = subprocess.run(
            ["pkill", "-9", "flite"],
            capture_output=True
        )
        if result.returncode == 0:
            stopped.append("flite")
        
        # Kill say (macOS TTS)
        if platform.system() == "Darwin":
            result = subprocess.run(
                ["pkill", "-9", "say"],
                capture_output=True
            )
            if result.returncode == 0:
                stopped.append("say")
        
        # Try to stop pyttsx3 by killing Python processes running it
        # This is more aggressive but necessary for TTS
        try:
            # Get list of Python processes
            ps_result = subprocess.run(
                ["ps", "aux"],
                capture_output=True,
                text=True
            )
            
            # Look for pyttsx3 or wakeword.py processes
            for line in ps_result.stdout.split('\n'):
                if 'pyttsx3' in line or 'wakeword.py' in line:
                    # Extract PID (second column)
                    parts = line.split()
                    if len(parts) > 1:
                        try:
                            pid = int(parts[1])
                            # Don't kill ourselves
                            if pid != os.getpid():
                                subprocess.run(["kill", "-9", str(pid)], capture_output=True)
                                stopped.append("TTS")
                                break
                        except (ValueError, IndexError):
                            pass
        except Exception:
            pass
        
        if stopped:
            # Remove duplicates and format
            stopped = list(set(stopped))
            return f"Stopped audio playback ({', '.join(stopped)})"
        else:
            return "No audio is currently playing"
    except Exception as e:
        return f"Error stopping audio: {str(e)}"
