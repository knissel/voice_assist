"""
YouTube Music integration for voice assistant.
Requires ytmusicapi and authentication setup.
"""
import os
import json
from ytmusicapi import YTMusic
from tools.audio import pause_media, clear_media_pause_state, is_media_paused_by_assistant

_active_headless_process = None


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

class YouTubeMusicManager:
    def __init__(self):
        self.ytmusic = YTMusic()

    def _extract_video_id_from_section(self, section):
        if not section:
            return None, None
        results = section.get("results") if isinstance(section, dict) else section
        if isinstance(results, list):
            for item in results:
                video_id = item.get("videoId")
                if video_id:
                    title = item.get("title")
                    artists = item.get("artists") or []
                    artist_name = artists[0].get("name") if artists else None
                    return video_id, {"title": title, "artist": artist_name}
        return None, None

    def _extract_artist_top_track(self, browse_id: str):
        try:
            artist = self.ytmusic.get_artist(browse_id)
        except Exception as e:
            return None, None, f"Artist lookup failed: {str(e)}"
        for key in ("songs", "topSongs", "videos"):
            video_id, meta = self._extract_video_id_from_section(artist.get(key))
            if video_id:
                return video_id, meta, None
        return None, None, None

    def _extract_album_track(self, browse_id: str):
        try:
            album = self.ytmusic.get_album(browse_id)
        except Exception as e:
            return None, None, f"Album lookup failed: {str(e)}"
        tracks = album.get("tracks") or []
        for track in tracks:
            video_id = track.get("videoId")
            if video_id:
                return video_id, {
                    "title": track.get("title"),
                    "artist": (track.get("artists") or [{}])[0].get("name")
                }, None
        return None, None, None
    
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
            video_id = result.get("videoId")
            browse_id = result.get("browseId")

            if search_type == "artists" and browse_id:
                artist_video_id, meta, error = self._extract_artist_top_track(browse_id)
                if error:
                    return {"error": error}
                if artist_video_id:
                    return {
                        "video_id": artist_video_id,
                        "url": f"https://music.youtube.com/watch?v={artist_video_id}",
                        "title": meta.get("title") if meta else result.get("title", "Unknown"),
                        "artist": meta.get("artist") if meta else result.get("title", "Unknown"),
                        "artist_url": f"https://music.youtube.com/channel/{browse_id}"
                    }
                return {
                    "artist_url": f"https://music.youtube.com/channel/{browse_id}",
                    "title": result.get("title", "Unknown"),
                    "artist": result.get("title", "Unknown")
                }

            if search_type == "albums" and browse_id:
                album_video_id, meta, error = self._extract_album_track(browse_id)
                if error:
                    return {"error": error}
                if album_video_id:
                    return {
                        "video_id": album_video_id,
                        "url": f"https://music.youtube.com/watch?v={album_video_id}",
                        "title": meta.get("title") if meta else result.get("title", "Unknown"),
                        "artist": meta.get("artist") if meta else "Unknown",
                        "album_url": f"https://music.youtube.com/browse/{browse_id}"
                    }

            if not video_id:
                return {"error": "Could not get video ID from search result"}

            return {
                "video_id": video_id,
                "url": f"https://music.youtube.com/watch?v={video_id}",
                "title": result.get("title", "Unknown"),
                "artist": result.get("artists", [{}])[0].get("name", "Unknown") if result.get("artists") else "Unknown"
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
    import shutil
    
    manager = YouTubeMusicManager()

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
        if content_type == "artist":
            artist_url = result.get("artist_url")
            if artist_url and not headless_mode:
                url = artist_url
            else:
                return "Failed to find a playable track for that artist."
        elif content_type == "album":
            album_url = result.get("album_url")
            if album_url and not headless_mode:
                url = album_url
            else:
                return "Failed to find a playable track for that album."
        else:
            return "Failed to get playback URL"
    
    # Check if headless mode is enabled (for Raspberry Pi)
    headless_mode = os.getenv("HEADLESS_PLAYBACK", "false").lower() == "true"
    
    try:
        if headless_mode:
            _stop_headless_playback()
            # Headless mode: stream audio directly using mpv or yt-dlp + ffplay.
            mpv_path = shutil.which("mpv")
            ytdlp_path = shutil.which("yt-dlp")
            ffplay_path = shutil.which("ffplay")

            if mpv_path:
                _active_headless_process = subprocess.Popen(
                    [mpv_path, "--no-video", "--really-quiet", url],
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
            
            # Look for pyttsx3 or run_assistant.py processes
            for line in ps_result.stdout.split('\n'):
                if 'pyttsx3' in line or 'run_assistant.py' in line:
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
