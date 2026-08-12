"""Accessible network-free learner package rendering."""

from __future__ import annotations

import zipfile
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from exam_prep_skill.models import LearnerExam, LearnerFlashcardDeck


def render_flashcards(deck: LearnerFlashcardDeck) -> str:
    """Render a searchable, LO-grouped offline flashcard deck."""
    payload = _safe_json(deck.model_dump_json())
    return _document(
        title=deck.title,
        app="flashcards",
        payload=payload,
        body="""
<header class="topbar"><div><p class="eyebrow">Offline study deck</p><h1 id="package-title"></h1></div><div class="progress-wrap"><span id="progress-label">0 / 0</span><div class="progress"><i id="progress-bar"></i></div></div></header>
<main class="deck-layout">
  <aside class="sidebar" aria-label="Study filters">
    <label for="objective-filter">Find a learning objective</label>
    <input id="objective-filter" type="search" placeholder="Search LO title">
    <div class="filter-actions"><button data-action="select-visible">Select visible</button><button data-action="clear-objectives">Clear</button></div>
    <div id="objective-list" class="objective-list"></div>
  </aside>
  <section class="study-area">
    <nav class="study-toolbar" aria-label="Card tools"><button data-action="export">Export progress</button><label class="file-button">Import progress<input id="import-progress" type="file" accept="application/json"></label><button data-action="reset">Reset</button></nav>
    <div class="stage">
      <button class="edge-arrow left" data-action="previous" aria-label="Previous card">&#8249;</button>
      <article id="study-card" class="study-card" tabindex="0" aria-live="polite"><p id="card-slot" class="eyebrow"></p><h2 id="card-prompt"></h2><div id="card-answer" class="answer" hidden></div><p id="card-source" class="source"></p><button class="primary" data-action="flip">Reveal answer</button><div id="ratings" class="ratings" hidden><button data-rating="again">Again</button><button data-rating="hard">Hard</button><button data-rating="good">Good</button><button data-rating="easy">Easy</button></div></article>
      <button class="edge-arrow right" data-action="next" aria-label="Next card">&#8250;</button>
    </div>
    <div><label for="card-queue">Jump to a card</label><select id="card-queue"></select></div>
  </section>
</main>""",
        script=FLASHCARD_SCRIPT,
    )


def render_mock_exam(exam: LearnerExam) -> str:
    """Render an offline mock exam with grading and explanations."""
    payload = _safe_json(exam.model_dump_json())
    return _document(
        title=exam.title,
        app="mock-exam",
        payload=payload,
        body="""
<header class="topbar"><div><p class="eyebrow">Offline mock exam</p><h1 id="package-title"></h1></div><div id="exam-timer" class="timer" aria-live="polite">00:00</div></header>
<main class="exam-layout"><aside><h2>Questions</h2><div id="question-navigator" class="question-grid"></div><button data-action="export">Export progress</button></aside><section class="exam-stage"><div class="question-meta"><span id="question-position"></span><button data-action="flag">Flag question</button></div><h2 id="question-prompt"></h2><fieldset id="choices"></fieldset><div id="answer-explanation" class="explanation" hidden><h3>Explanation</h3><p id="explanation-text"></p><p id="source-pages"></p></div><nav class="exam-actions"><button data-action="previous">Previous</button><button data-action="next">Next</button><button class="primary" data-action="submit">Submit exam</button></nav><div id="score" class="score" aria-live="polite"></div></section></main>""",
        script=EXAM_SCRIPT,
    )


def write_learner_zip(destination: Path, filename: str, html: str) -> None:
    """Write exactly one learner-facing HTML file to a package ZIP."""
    if Path(filename).suffix.casefold() != ".html" or Path(filename).name != filename:
        msg = "Learner package output must be one top-level HTML file"
        raise ValueError(msg)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(filename, html.encode())


def _document(*, title: str, app: str, payload: str, body: str, script: str) -> str:
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{escape(title)}</title><style>{BASE_CSS}</style></head><body data-app="{app}">{body}<script id="package-data" type="application/json">{payload}</script><script>{script}</script></body></html>"""


def _safe_json(value: str) -> str:
    return (
        value.replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("/script", "\\/script")
    )


BASE_CSS = """
:root{color-scheme:light;--ink:#172025;--muted:#617078;--line:#d9e0dd;--paper:#fbfcfb;--teal:#176c63;--teal-soft:#e7f2ef;--amber:#d9922e;--shadow:0 20px 60px rgba(20,40,37,.10)}*{box-sizing:border-box}body{margin:0;background:#f3f5f3;color:var(--ink);font:16px/1.5 ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;letter-spacing:0}button,input,select{font:inherit}button,.file-button{border:1px solid var(--line);background:white;color:var(--ink);padding:.7rem 1rem;border-radius:6px;cursor:pointer}button:focus-visible,input:focus-visible,select:focus-visible,.study-card:focus-visible{outline:3px solid #61b7ac;outline-offset:3px}.primary{background:var(--teal);color:white;border-color:var(--teal)}.topbar{display:flex;align-items:center;justify-content:space-between;gap:2rem;padding:1.25rem clamp(1rem,4vw,4rem);background:white;border-bottom:1px solid var(--line)}h1,h2,h3,p{margin-top:0}h1{font-size:clamp(1.45rem,2.4vw,2.25rem);margin-bottom:0}.eyebrow{color:var(--teal);font-size:.78rem;font-weight:800;text-transform:uppercase;letter-spacing:.08em}.progress-wrap{min-width:180px}.progress{height:8px;background:#e4ebe8;border-radius:99px;overflow:hidden}.progress i{display:block;height:100%;background:var(--teal);width:0}.deck-layout,.exam-layout{display:grid;grid-template-columns:minmax(250px,320px) minmax(0,1fr);min-height:calc(100vh - 100px)}.sidebar,.exam-layout>aside{padding:1.5rem;background:white;border-right:1px solid var(--line);overflow:auto}.sidebar input{width:100%;padding:.75rem;border:1px solid var(--line);border-radius:6px}.filter-actions,.study-toolbar,.exam-actions,.ratings{display:flex;flex-wrap:wrap;gap:.6rem;margin:1rem 0}.objective-list{display:grid;gap:.6rem;max-height:60vh;overflow:auto}.objective-item{display:grid;grid-template-columns:auto 1fr;gap:.6rem;padding:.7rem;border:1px solid var(--line);border-radius:6px}.study-area,.exam-stage{padding:clamp(1rem,3vw,3rem);min-width:0}.study-toolbar{justify-content:flex-end}.file-button input{position:absolute;opacity:0;pointer-events:none}.stage{display:grid;grid-template-columns:52px minmax(0,1fr) 52px;align-items:stretch;gap:1rem}.edge-arrow{background:transparent;border:0;font-size:3.5rem;color:var(--teal);padding:0}.study-card{min-height:420px;background:var(--paper);border:1px solid #b8d8d2;border-radius:8px;box-shadow:var(--shadow);padding:clamp(1.5rem,5vw,5rem);display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center}.study-card h2{font-size:clamp(1.75rem,4vw,3.3rem);max-width:22ch}.answer{font-size:1.2rem;max-width:60ch}.source{color:var(--muted)}#card-queue{width:100%;padding:.75rem;margin-top:.4rem}.timer{font-size:1.4rem;font-weight:800;font-variant-numeric:tabular-nums}.question-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:.5rem;margin-bottom:1.5rem}.question-grid button{padding:.6rem}.question-grid button.current{background:var(--teal);color:white}.question-grid button.flagged{border-color:var(--amber)}.question-meta{display:flex;justify-content:space-between;gap:1rem}.exam-stage{max-width:960px}.exam-stage fieldset{border:0;padding:0;display:grid;gap:.8rem}.choice{display:flex;gap:.75rem;padding:1rem;border:1px solid var(--line);border-radius:6px;background:white}.explanation,.score{margin-top:1.5rem;padding:1rem;background:var(--teal-soft);border-left:4px solid var(--teal)}@media(max-width:760px){.topbar{align-items:flex-start}.deck-layout,.exam-layout{grid-template-columns:1fr}.sidebar,.exam-layout>aside{border-right:0;border-bottom:1px solid var(--line);max-height:320px}.stage{grid-template-columns:40px minmax(0,1fr) 40px;gap:.25rem}.study-card{min-height:390px;padding:1.5rem}.study-card h2{font-size:1.8rem}.progress-wrap{min-width:120px}.question-grid{grid-template-columns:repeat(8,1fr)}}
"""


FLASHCARD_SCRIPT = """
const payload=JSON.parse(document.getElementById('package-data').textContent);const key='exam-prep:'+payload.package_id;const saved=JSON.parse(localStorage.getItem(key)||'{}');const state={index:saved.index||0,revealed:false,selected:saved.selected||[...new Set(payload.cards.map(c=>c.objective_id))],ratings:saved.ratings||{}};const $=id=>document.getElementById(id);$('package-title').textContent=payload.title;const objectives=[...new Map(payload.cards.map(c=>[c.objective_id,{id:c.objective_id,code:c.objective_code,title:c.objective_title}])).values()];let visibleObjectives=objectives;function cards(){return payload.cards.filter(c=>state.selected.includes(c.objective_id))}function persist(){localStorage.setItem(key,JSON.stringify({index:state.index,selected:state.selected,ratings:state.ratings}))}function renderObjectives(){const host=$('objective-list');host.innerHTML='';visibleObjectives.forEach(o=>{const label=document.createElement('label');label.className='objective-item';const input=document.createElement('input');input.type='checkbox';input.checked=state.selected.includes(o.id);input.addEventListener('change',()=>{state.selected=input.checked?[...state.selected,o.id]:state.selected.filter(id=>id!==o.id);state.index=0;persist();render()});const span=document.createElement('span');span.textContent=o.code+' · '+o.title;label.append(input,span);host.append(label)})}function render(){const list=cards();if(!list.length){$('card-prompt').textContent='Select at least one learning objective';$('card-answer').hidden=true;$('progress-label').textContent='0 / 0';$('card-queue').innerHTML='';return}state.index=Math.max(0,Math.min(state.index,list.length-1));const c=list[state.index];$('card-slot').textContent=c.objective_code+' · '+c.objective_title+' · '+c.slot;$('card-prompt').textContent=c.prompt;$('card-answer').textContent=c.answer;$('card-answer').hidden=!state.revealed;$('ratings').hidden=!state.revealed;$('card-source').textContent=c.module_title+' · page '+c.source_pages.join(', ');$('progress-label').textContent=(state.index+1)+' / '+list.length;$('progress-bar').style.width=((state.index+1)/list.length*100)+'%';$('card-queue').innerHTML='';list.forEach((card,i)=>{const option=document.createElement('option');option.value=i;option.selected=i===state.index;option.textContent=(i+1)+'. '+card.objective_code+' — '+card.prompt;$('card-queue').append(option)});persist()}function move(delta){const list=cards();if(!list.length)return;state.index=(state.index+delta+list.length)%list.length;state.revealed=false;render()}function exportProgress(){const blob=new Blob([JSON.stringify({package_id:payload.package_id,kind:'flashcards',progress:{index:state.index,selected:state.selected,ratings:state.ratings}},null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='flashcard-progress.json';a.click();URL.revokeObjectURL(a.href)}document.querySelectorAll('[data-action]').forEach(b=>b.addEventListener('click',()=>{const action=b.dataset.action;if(action==='previous')move(-1);if(action==='next')move(1);if(action==='flip'){state.revealed=!state.revealed;render()}if(action==='select-visible'){state.selected=[...new Set([...state.selected,...visibleObjectives.map(o=>o.id)])];renderObjectives();render()}if(action==='clear-objectives'){state.selected=[];state.index=0;renderObjectives();render()}if(action==='export')exportProgress();if(action==='reset'&&confirm('Reset saved progress?')){localStorage.removeItem(key);location.reload()}}));document.querySelectorAll('[data-rating]').forEach(b=>b.addEventListener('click',()=>{const c=cards()[state.index];state.ratings[c.card_id]=b.dataset.rating;move(1)}));$('objective-filter').addEventListener('input',e=>{const q=e.target.value.toLowerCase();visibleObjectives=objectives.filter(o=>(o.code+' '+o.title).toLowerCase().includes(q));renderObjectives()});$('card-queue').addEventListener('change',e=>{state.index=Number(e.target.value);state.revealed=false;render()});$('import-progress').addEventListener('change',e=>{const file=e.target.files[0];if(!file)return;const reader=new FileReader();reader.onload=()=>{const data=JSON.parse(reader.result);if(data.package_id!==payload.package_id){alert('This progress file belongs to another deck.');return}Object.assign(state,data.progress,{revealed:false});renderObjectives();render()};reader.readAsText(file)});document.addEventListener('keydown',e=>{if(e.key==='ArrowLeft')move(-1);if(e.key==='ArrowRight')move(1);if(e.key===' '){e.preventDefault();state.revealed=!state.revealed;render()}});renderObjectives();render();
"""


EXAM_SCRIPT = """
const payload=JSON.parse(document.getElementById('package-data').textContent);const key='exam-prep:'+payload.package_id;const saved=JSON.parse(localStorage.getItem(key)||'{}');const state={index:saved.index||0,answers:saved.answers||{},flags:saved.flags||[],remaining:saved.remaining??payload.duration_minutes*60,submitted:saved.submitted||false};const $=id=>document.getElementById(id);$('package-title').textContent=payload.title;function persist(){localStorage.setItem(key,JSON.stringify(state))}function renderNav(){const host=$('question-navigator');host.innerHTML='';payload.questions.forEach((q,i)=>{const b=document.createElement('button');b.textContent=i+1;b.className=(i===state.index?'current ':'')+(state.flags.includes(q.question_id)?'flagged':'');b.setAttribute('aria-label','Question '+(i+1));b.addEventListener('click',()=>{state.index=i;render()});host.append(b)})}function render(){const q=payload.questions[state.index];$('question-position').textContent='Question '+(state.index+1)+' of '+payload.questions.length+' · '+q.objective_code;$('question-prompt').textContent=q.prompt;const host=$('choices');host.innerHTML='';q.choices.forEach((text,i)=>{const id=q.question_id+'-'+i;const label=document.createElement('label');label.className='choice';const input=document.createElement('input');input.type='radio';input.name=q.question_id;input.value=String.fromCharCode(65+i);input.id=id;input.checked=state.answers[q.question_id]===input.value;input.disabled=state.submitted;input.addEventListener('change',()=>{state.answers[q.question_id]=input.value;persist();renderNav()});const span=document.createElement('span');span.textContent=String.fromCharCode(65+i)+'. '+text;label.append(input,span);host.append(label)});$('answer-explanation').hidden=!state.submitted;$('explanation-text').textContent=q.explanation;$('source-pages').textContent='Source: '+q.objective_title+' · page '+q.source_pages.join(', ');renderNav();persist()}function move(delta){state.index=Math.max(0,Math.min(payload.questions.length-1,state.index+delta));render()}function submitExam(){if(!confirm('Submit and grade this exam?'))return;state.submitted=true;const correct=payload.questions.filter(q=>state.answers[q.question_id]===q.correct_choice).length;$('score').textContent='Score: '+correct+' / '+payload.questions.length+' ('+Math.round(correct/payload.questions.length*100)+'%)';render()}function exportProgress(){const blob=new Blob([JSON.stringify({package_id:payload.package_id,kind:'mock_exam',progress:state},null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='mock-exam-progress.json';a.click();URL.revokeObjectURL(a.href)}document.querySelectorAll('[data-action]').forEach(b=>b.addEventListener('click',()=>{const action=b.dataset.action;if(action==='previous')move(-1);if(action==='next')move(1);if(action==='flag'){const id=payload.questions[state.index].question_id;state.flags=state.flags.includes(id)?state.flags.filter(x=>x!==id):[...state.flags,id];render()}if(action==='submit')submitExam();if(action==='export')exportProgress()}));setInterval(()=>{if(state.remaining>0&&!state.submitted){state.remaining--;persist()}const m=Math.floor(state.remaining/60);const s=state.remaining%60;$('exam-timer').textContent=String(m).padStart(2,'0')+':'+String(s).padStart(2,'0')},1000);render();
"""
