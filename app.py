import json
import re
from io import BytesIO
from hashlib import md5, sha1

import streamlit as st
from openai import OpenAI
from st_click_detector import click_detector
from streamlit_player import st_player

from streamlit_env import Env

AUDIO_TRANSCRIBE_MODEL = "whisper-1"
MAX_SIZE = 25_000_000
# MAX_SIZE = 250
env = Env('.env')


store_keys = {
    'segments': ['id', 'seek', 'start', 'end', 'text']
}


def write_html(html: str):
    st.write(html, unsafe_allow_html=True)


def format_tm(tm):
    tm = int(tm)
    parts = [tm // 3600, (tm % 3600) // 60, tm % 60]
    for _ in range(2):
        parts = parts[1:] if not parts[0] else parts
    return ':'.join([f"{nr:02d}" for nr in parts])


def transcribe_audio(_openai_client, audio_bytes: bytes, word_timestamp=False) -> str:
    audio_file = BytesIO(audio_bytes)
    audio_file.name = "audio.mp3"
    timestamp_granularities = ["segment"]
    if word_timestamp:
        timestamp_granularities = ['word'] + timestamp_granularities
    transcript = _openai_client.audio.transcriptions.create(
        file=audio_file,
        model=AUDIO_TRANSCRIBE_MODEL,
        response_format="verbose_json",
        timestamp_granularities=timestamp_granularities,
    )
    return transcript.text, getattr(transcript, 'words', []), getattr(transcript, 'segments', [])


def get_audio_hash(audio_bytes):
    _hsh = md5(audio_bytes).hexdigest()
    return _hsh


def set_video_offset(offset=0):
    st.session_state.video_offset = offset


def get_segment_lines(segments, pattern):
    _html_lines = []
    for _nr, _line in enumerate(segments):
        line2 = dict(_line)
        for k3 in ['start', 'end']:
            line2[f'{k3}2'] = format_tm(line2[k3])
        s_line = {'time': int(line2['start']), 'id': _nr}
        line2['int_tm'] = s_line['time']
        s_line['html'] = pattern % ({'id': _nr} | line2)
        _html_lines.append(s_line)
    return _html_lines


@st.cache_resource
def get_openai_client():
    return OpenAI(api_key=st.session_state.api_key)


session_defaults = [('api_key', None), ('file_key', 1), ('file_hash', None), ('text', ''), ('words', ''),
                    ('segments', ''), ('uploaded_filename', ''), ('video_offset', 0), ('clicked', 1)]

for sess_k, def_v in session_defaults:
    if sess_k not in st.session_state:
        st.session_state[sess_k] = def_v


can_process = True
allowed_emails = env.get('ALLOWED_EMAILS')
if allowed_emails:
    can_process = False
    email = st.text_input('Podaj swój email')
    allowed_emails = allowed_emails.split(',')
    if email in allowed_emails:
        can_process = True

if not st.session_state.api_key:
    api_key = st.text_input('Podaj Hasło', type='password')
    if api_key:
        hsh = env.get('PWD_HASH')
        salt = env.get('PWD_SALT')
        env_api_key = env.get('OPENAI_API_KEY')
        if hsh and salt and env_api_key:
            if sha1(f'{salt} {api_key}'.encode()).hexdigest() == hsh:
                api_key = env_api_key
        st.session_state.api_key = api_key
        st.rerun()
else:
    f_key = f"k_{st.session_state['file_key']}"

    tab1, tab2 = st.tabs(['Przetwarzanie audio', 'prezentacja'])

    with tab1:
        uploaded_file = st.file_uploader('Wczytaj plik mp3', type=['mp3'], accept_multiple_files=False, key=f_key,
                                         help='Max 25MB')
        fetch_word_timestamps = st.checkbox('Pobierz timestamp wszystkich wyrazów')
        if can_process and uploaded_file:
            bts = BytesIO(uploaded_file.getvalue()).getvalue()
            is_fine = len(bts) <= MAX_SIZE
            if not is_fine:
                st.error(f':red[Plik nie może być większy niż {MAX_SIZE}]')
                ok_error = st.button('OK', type='primary')
                if ok_error:
                    st.session_state.file_key = st.session_state['file_key'] + 1
                    st.rerun()
            else:
                uploaded_filename = uploaded_file.name
                file_hash = get_audio_hash(bts)
                openai_client = get_openai_client()
                if file_hash != st.session_state.file_hash:
                    st.session_state.file_hash = file_hash
                    st.session_state.uploaded_filename = uploaded_filename
                    data = transcribe_audio(openai_client, bts, word_timestamp=fetch_word_timestamps)
                    for nr, k in enumerate(['text', 'words', 'segments']):
                        dt = data[nr]
                        if dt and k in store_keys and isinstance(dt, list) and isinstance(dt[0], dict):
                            dt = [{k2: v2 for k2, v2 in itm.items() if k2 in store_keys[k]} for itm in dt]
                        st.session_state[k] = dt
                for k in ['text', 'words', 'segments']:
                    st.header(k)
                    st.write(st.session_state[k])
                    if st.session_state[k]:
                        ext = 'txt' if k == 'text' else 'json'
                        dt2store = json.dumps(st.session_state[k]) if k in [
                            'words', 'segments'] else str(st.session_state[k])
                        st.download_button(label='Pobierz', data=dt2store,
                                           file_name=f'{st.session_state.uploaded_filename}_{k}.{ext}')
                download_html = st.button('Spreparuj html', key=f'html_{f_key}', help='Pobierz tekst jako html')
                if download_html:
                    p = '<p id="p_%(id)s" title="start: %(start2)s, end: %(end2)s" class="txt_line">'\
                        '<span class="timestamp" data-tm="%(int_tm)s">(%(start2)s - %(end2)s)</span>'\
                        '\n%(text)s\n'\
                        '</p>'
                    html_lines = get_segment_lines(st.session_state['segments'], p)
                    jquery = open('jquery.js').read()
                    tpl = open('html_template.txt', encoding='utf8').read()
                    full_html = tpl % {'title': st.session_state.uploaded_filename, 'body': '\n'.join(html_lines),
                                    'jquery': jquery}
                    st.download_button(label='Pobierz html', data=full_html,
                                       file_name=f'{st.session_state.uploaded_filename}.html')
                
                reset_all = st.button('Zresetuj', key=f'btn_{f_key}', type='primary',
                                      help='Umożliwia wczytanie innego pliku')
                if reset_all:
                    st.session_state.file_key = st.session_state['file_key'] + 1
                    for k in ['text', 'words', 'segments']:
                        st.session_state[k] = ''
                        st.rerun()
    with tab2:
        if not st.session_state.segments:
            load_segments = st.file_uploader('Załaduj json z danymi "segments"', key=f'segm_{f_key}', type=['json'])
            if load_segments:
                st.session_state.segments = json.loads(load_segments.getvalue())
                st.rerun()
        else:
            url = st.text_input('URL z video na YT', on_change=set_video_offset)
            yt_src = ''
            if url:
                if url and re.match(r'^https?://(www\.youtube\.com|youtu\.be)/[a-zA-Z0-9_.+=,#\/?&%-]+$', url):
                    key_yt = re.match(r'^https?://[^/]+/embed/([a-zA-Z0-9_.-]+)', url)
                    key_yt = key_yt or re.match(r'^https?://[^?]+.*?[?&]v=([a-zA-Z0-9_.-]+)', url)
                    key_yt = key_yt[1] if key_yt else None
                    start_yt = st.session_state.video_offset or 0
                    yt_src = f'http://www.youtube.com/embed/{key_yt}&start={start_yt}' if key_yt else ''
            col1, col2 = st.columns(2)
            clicked = {}
            with col1:
                htm_p = '<p title="start: %(start2)s, end: %(end2)s">%(text)s</p>'
                html_lines = get_segment_lines(st.session_state['segments'], htm_p)
                for line in html_lines:
                    write_html(line['html'])
                    if yt_src:
                        c_key = st.session_state['clicked']
                        clicked[line['id']] = click_detector(
                            f'''<a href="#" id="clck_{line['id']}">-&gt;</a>''',
                            key=f"goto_{c_key}_{line['id']}")
                        if clicked[line['id']]:
                            st.session_state.clicked = st.session_state['clicked'] + 1
                            set_video_offset(offset=line['time'])
                            st.rerun()
            with col2:
                if yt_src:
                    st_player(yt_src)
