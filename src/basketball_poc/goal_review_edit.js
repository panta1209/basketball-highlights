// Manual decisions are separate from immutable automatic candidates.
let manualEvents=[],rangeOverrides={},manualWidgets=[],editing=null;
const editEl=id=>document.getElementById(id);
const duration=DATA.manifest.source.duration_seconds;
function validInterval(start,time,end){return [start,time,end].every(Number.isFinite)&&0<=start&&start<=time&&time<end&&end<=duration;}
function clipInterval(c){
 const override=rangeOverrides[c.id];
 if(override)return {start:override.start_seconds,end:override.end_seconds};
 if(c.id.startsWith('manual-'))return {start:c.start_seconds,end:c.end_seconds};
 return DATA.preview.segments.find(s=>s.id===c.id);
}
function restoreEdits(saved){
 const manual=saved.manual_events||[],ranges=saved.segment_overrides||{},seen=new Set();
 if(!Array.isArray(manual)||manual.length>1000||!ranges||typeof ranges!=='object'||Array.isArray(ranges))throw Error('草稿格式错误');
 for(const e of manual){if(!e||!/^manual-[a-zA-Z0-9-]{1,64}$/.test(e.id)||seen.has(e.id)||!validInterval(e.start_seconds,e.time_seconds,e.end_seconds))throw Error('补充片段已损坏');seen.add(e.id);}
 for(const [id,r] of Object.entries(ranges)){const c=candidates.find(c=>c.id===id);if(!c||!r||!validInterval(r.start_seconds,c.time_seconds,r.end_seconds))throw Error('片段范围已损坏');}
 manualEvents=manual;rangeOverrides=ranges;
}
function editButton(text,action){const b=document.createElement('button');b.type='button';b.textContent=text;b.dataset.editControl='';b.addEventListener('click',action);return b;}
function openEditor(c=null,refresh=null){
 document.querySelectorAll('video').forEach(v=>v.pause());
 editing={candidate:c,refresh};const manual=!c||c.id.startsWith('manual-');
 editEl('editor-title').textContent=c?(manual?'修改补充进球':'调整片段范围'):'补充进球';
 const time=c?.time_seconds??Math.min(4,duration/2),interval=c?clipInterval(c):{start:Math.max(0,time-4),end:Math.min(duration,time+2)};
 editEl('edit-time').value=time.toFixed(2);editEl('edit-time').readOnly=!manual;
 editEl('use-time').hidden=!manual;
 editEl('edit-start').value=interval.start.toFixed(2);editEl('edit-end').value=interval.end.toFixed(2);
 editEl('edit-error').textContent='';
 const video=editEl('editor-video');
 if(!video.getAttribute('src')){video.src=DATA.preview.url;video.load();}
 if(video.readyState)video.currentTime=time;else video.addEventListener('loadedmetadata',()=>video.currentTime=time,{once:true});
 editEl('goal-editor').showModal();
}
function renderManual(){
 manualWidgets.forEach(w=>w.destroy());manualWidgets=[];editEl('manual-goals').replaceChildren();
 editEl('manual-section').hidden=!manualEvents.length;
 manualEvents.forEach((e,i)=>{
  const card=document.createElement('article');card.className='card';
  const widget=segmentPlayer(document.createElement('video'),e,`补充进球 ${i+1}`);manualWidgets.push(widget);card.append(widget.element);
  const body=document.createElement('div');body.className='body';const title=document.createElement('h3');title.textContent=`补充进球 ${i+1} · ${stamp(e.time_seconds)}`;
  const note=document.createElement('p');note.textContent='手动补充 · 已加入集锦';
  const actions=document.createElement('div');actions.className='actions';
  actions.append(editButton('修改片段',()=>openEditor(e)),editButton('移除补充进球',()=>{manualEvents=manualEvents.filter(x=>x.id!==e.id);persist();invalidate();renderManual();update();}));
  body.append(title,note,actions);card.append(body);editEl('manual-goals').append(card);
 });
}
editEl('add-goal').addEventListener('click',()=>openEditor());
editEl('cancel-edit').addEventListener('click',()=>editEl('goal-editor').close());
editEl('goal-editor').addEventListener('close',()=>editEl('editor-video').pause());
for(const field of ['time','start','end'])editEl('use-'+field).addEventListener('click',()=>{
 const time=Math.round(editEl('editor-video').currentTime*100)/100;
 editEl('edit-'+field).value=time;
 if(field==='time'&&!editing.candidate){editEl('edit-start').value=Math.max(0,time-4).toFixed(2);editEl('edit-end').value=Math.min(duration,time+2).toFixed(2);}
});
editEl('goal-edit-form').addEventListener('submit',event=>{
 event.preventDefault();if(busy||playerBusy)return;
 const time=Number(editEl('edit-time').value),start=Number(editEl('edit-start').value),end=Number(editEl('edit-end').value);
 if(!validInterval(start,time,end)){editEl('edit-error').textContent='开始时间不能晚于进球时刻；结束时间必须晚于进球时刻，且不能超出视频。';return;}
 const c=editing.candidate;
 if(!c||c.id.startsWith('manual-')){
  const entry={id:c?.id||'manual-'+crypto.randomUUID(),time_seconds:time,start_seconds:start,end_seconds:end};
  if(c)manualEvents=manualEvents.map(e=>e.id===c.id?entry:e);else manualEvents.push(entry);
  manualEvents.sort((a,b)=>a.time_seconds-b.time_seconds);renderManual();
 }else{rangeOverrides[c.id]={start_seconds:start,end_seconds:end};editing.refresh?.();}
 persist();invalidate();update();editEl('goal-editor').close();
});
