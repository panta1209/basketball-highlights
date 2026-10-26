let playerState=null,playerDraft=null,playerDirty=false,playerExportStatus=null,playerSubmitting=false;
let playerWidgets=[];
const playerEl=id=>document.getElementById(id);
const candidateMap=()=>new Map((playerState?.automatic?.events||[]).flatMap(e=>e.candidates.map(c=>[c.id,c])));
const samePlayerSelection=()=>playerState&&signature(playerState.selection?.choices)===signature(choices);
const playerRequest=()=>({analysis_id:playerState.analysis_id,selection:selection(),revision:playerDraft.revision});
const newPlayerId=()=> 'p-'+crypto.randomUUID().replaceAll('-','').slice(0,12);
function showReviewScreen(screen,navigate=true){
 const players=PLAYERS_ENABLED&&screen==='players';
 playerEl('goal-screen').hidden=players;playerEl('player-screen').hidden=!players;
 playerEl('goal-footer').hidden=players;playerEl('player-footer').hidden=!players;
 playerEl('screen-step').textContent=players?'第 2 步 / 共 2 步':'第 1 步 / 共 2 步';
 playerEl('screen-step').hidden=!PLAYERS_ENABLED;
 playerEl('screen-title').textContent=players?'核对投篮人，选人导出':'确认进球';
 if(!PLAYERS_ENABLED){
  playerEl('screen-title').textContent='确认进球并生成集锦';
  playerEl('goal-screen').querySelector('p').textContent='确认要保留的进球，然后生成原声集锦。';
 }
 document.title=players?'篮球集锦 · 人物核对':'篮球集锦 · 确认进球';
 document.querySelectorAll('video').forEach(video=>video.pause());
 const hash=players?'#players':'#goals';
 if(!PLAYERS_ENABLED&&(location.hash==='#players'||location.hash==='#player-step'))history.replaceState(null,'','#goals');
 else if(navigate&&location.hash!==hash)history.pushState(null,'',hash);
 if(navigate){window.scrollTo(0,0);playerEl('screen-title').focus({preventScroll:true});}
}
window.addEventListener('hashchange',()=>showReviewScreen(
 location.hash==='#players'||location.hash==='#player-step'?'players':'goals',false));
async function playerPost(endpoint,value){
 const response=await fetch('api/players/'+endpoint,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(value)});
 const result=await response.json();if(!response.ok)throw Error(result.error||'操作未完成，请重试。');return result;
}
function playerButton(text,action){const button=document.createElement('button');button.textContent=text;button.addEventListener('click',action);return button;}
function playerPicture(id){
 const item=candidateMap().get(id),box=document.createElement('div');box.className='portrait';
 if(item){const img=document.createElement('img');img.src=item.thumbnail_url;img.alt='人物代表画面';img.loading='lazy';box.append(img);}
 else{box.textContent='暂无人物图';box.classList.add('empty');}return box;
}
function playerChanged(){
 playerDirty=true;playerEl('player-montage').hidden=true;playerEl('player-record').hidden=true;
 playerEl('player-message').textContent='人物归属尚未保存。核对完后点击“保存人物归属”。';updatePlayerExport();
}
function playerPending(){return playerState.automatic.events.filter(e=>!playerDraft.assignments[e.id]&&!playerDraft.skipped.includes(e.id)).length;}
function updatePlayerExport(){
 const ready=playerState?.status==='completed'&&samePlayerSelection()&&playerDraft;
 playerEl('save-players').disabled=!ready||playerSubmitting||busy||playerBusy;
 playerEl('new-player').disabled=!ready||playerSubmitting||busy||playerBusy;
 const count=ready?Object.values(playerDraft.assignments).filter(id=>id===playerEl('export-player').value).length:0;
 playerEl('export-one').disabled=!ready||playerDirty||playerPending()>0||playerDraft.actor==='automatic_suggestion'||!count||busy||playerBusy||playerSubmitting;
}
function checkPlayerSelection(){
 if(!playerState||playerState.status!=='completed')return;
 const wasChanged=playerEl('player-step').classList.contains('outdated');
 const changed=!samePlayerSelection();
 playerEl('player-step').classList.toggle('outdated',changed);
 if(changed){playerEl('player-message').textContent='进球选择已变化，请返回进球复核，再点击“下一步：匹配投篮人”更新。';
  playerEl('player-events').hidden=true;playerEl('player-groups').hidden=true;playerEl('player-merges').hidden=true;
 }else{playerEl('player-events').hidden=false;playerEl('player-groups').hidden=false;
  playerEl('player-merges').hidden=!playerEl('player-merges').childNodes.length;
  if(wasChanged)playerEl('player-message').textContent=playerDirty?'人物归属尚未保存。':`${playerState.automatic.events.length} 个进球 · ${playerPending()} 个待指定。人物归属已恢复。`;}
 updatePlayerExport();
}
function addPlayer(representative=null){
 const numbers=playerDraft.players.map(p=>/^人物 (\d+)$/.exec(p.name)?.[1]).filter(Boolean).map(Number);
 const player={id:newPlayerId(),name:`人物 ${Math.max(0,...numbers)+1}`,representative};playerDraft.players.push(player);return player;
}
function ownedGoals(player){return playerState.automatic.events.filter(event=>playerDraft.assignments[event.id]===player.id);}
function playerLabel(player){
 const numbers=ownedGoals(player).map(event=>playerState.automatic.events.indexOf(event)+1);
 return `${player.name}（${numbers.length?'第 '+numbers.join('、')+' 球':'暂无进球'}）`;
}
function setGoalOwner(eventId,value){
 const previous=playerDraft.assignments[eventId];
 const next=value==='__skip__'||!value?null:value;
 if(next===previous && playerDraft.skipped.includes(eventId)===(value==='__skip__'))return;
 playerDraft.assignments[eventId]=next;
 playerDraft.chosen_candidates[eventId]=null;
 playerDraft.skipped=playerDraft.skipped.filter(id=>id!==eventId);
 if(value==='__skip__')playerDraft.skipped.push(eventId);
 if(previous){
  const owner=playerDraft.players.find(player=>player.id===previous);
  if(owner){
   const remaining=ownedGoals(owner);
   if(!remaining.length)playerDraft.players=playerDraft.players.filter(player=>player.id!==previous);
   else if(!remaining.some(event=>event.candidates.some(candidate=>candidate.id===owner.representative)))
    owner.representative=remaining.map(event=>playerDraft.chosen_candidates[event.id])
      .find(id=>id&&candidateMap().has(id))||null;
  }
 }
 playerChanged();renderPlayers();
}
function mergePlayers(from,to){
 if(from===to)return;
 for(const [event,id] of Object.entries(playerDraft.assignments))if(id===from)playerDraft.assignments[event]=to;
 playerDraft.players=playerDraft.players.filter(p=>p.id!==from);playerChanged();renderPlayers();
}
function currentMergeSuggestions(){
 const candidates=candidateMap(),suggestions=[];
 const threshold=playerState?.automatic?.settings?.merge_suggest_threshold??.12;
 function distance(a,b){
  const part=key=>Math.sqrt(a[key].reduce((sum,value,i)=>sum+(Math.sqrt(value)-Math.sqrt(b[key][i]))**2,0)/a[key].length);
  const color=part('color'),texture=part('texture');return {score:.7*color+.3*texture,texture};
 }
 for(let i=0;i<playerDraft.players.length;i++)for(let j=i+1;j<playerDraft.players.length;j++){
  const left=playerDraft.players[i],right=playerDraft.players[j],a=candidates.get(left.representative),b=candidates.get(right.representative);
  if(!a||!b)continue;
  if(a.views.some(x=>b.views.some(y=>x.frame===y.frame&&Math.abs(x.box[0]-y.box[0])+Math.abs(x.box[1]-y.box[1])>20)))continue;
  const values=a.views.flatMap(x=>b.views.map(y=>distance(x.feature,y.feature))).sort((x,y)=>x.score-y.score);
  const match=values[Math.floor(values.length/2)];
  if(match&&match.score<threshold&&match.texture<.15)suggestions.push({left:left.id,right:right.id,score:match.score});
 }
 return suggestions.sort((a,b)=>a.score-b.score).slice(0,4);
}
function renderPlayers(){
 if(!playerState||playerState.status!=='completed')return;
 playerEl('player-step').hidden=false;
 const groups=playerEl('player-groups'),events=playerEl('player-events'),merges=playerEl('player-merges');
 playerWidgets.forEach(widget=>widget.destroy());playerWidgets=[];
 groups.replaceChildren();events.replaceChildren();merges.replaceChildren();
 const previous=playerEl('export-player').value;playerEl('export-player').replaceChildren();
 for(const player of playerDraft.players){
  const goals=ownedGoals(player),count=goals.length;
  const card=document.createElement('article');card.className='card person-group';card.append(playerPicture(player.representative));
  const body=document.createElement('div');body.className='body';const input=document.createElement('input');input.value=player.name;input.maxLength=60;input.setAttribute('aria-label',player.name+' 名称');
  input.addEventListener('input',()=>{player.name=input.value;playerChanged();});input.addEventListener('change',renderPlayers);
  const number=document.createElement('p');number.textContent=`已归入 ${count} 个进球`;body.append(input,number);
  if(playerDraft.players.length>1)for(const goal of goals){
   const row=document.createElement('div');row.className='goal-transfer';
   const ordinal=playerState.automatic.events.indexOf(goal)+1;
   const label=document.createElement('span');label.textContent=`第 ${ordinal} 球 · ${stamp(goal.time_seconds)}`;
   const select=document.createElement('select');select.setAttribute('aria-label',`第 ${ordinal} 球改给谁`);
   select.add(new Option('改给另一位人物',''));
   for(const other of playerDraft.players.filter(other=>other.id!==player.id))select.add(new Option(playerLabel(other),other.id));
   const button=playerButton('只改这一球',()=>setGoalOwner(goal.id,select.value));button.disabled=true;
   select.addEventListener('change',()=>{button.disabled=!select.value;});
   row.append(label,select,button);body.append(row);
  }
  if(playerDraft.players.length>1){
   const details=document.createElement('details');details.className='group-merge';
   const summary=document.createElement('summary');summary.textContent='合并整组（会移动本组全部进球）';details.append(summary);
   const select=document.createElement('select');select.setAttribute('aria-label',playerLabel(player)+' 合并目标');
   for(const other of playerDraft.players.filter(other=>other.id!==player.id))select.add(new Option(playerLabel(other),other.id));
   details.append(select,playerButton('合并整组',()=>mergePlayers(player.id,select.value)));body.append(details);
  }
  card.append(body);groups.append(card);
  playerEl('export-player').add(new Option(playerLabel(player),player.id));
 }
 if(playerDraft.players.some(p=>p.id===previous))playerEl('export-player').value=previous;
 for(const suggestion of currentMergeSuggestions()){
  const left=playerDraft.players.find(p=>p.id===suggestion.left),right=playerDraft.players.find(p=>p.id===suggestion.right);
  if(left&&right)merges.append(playerButton(`${playerLabel(left)} 与 ${playerLabel(right)} 外观相似，合并整组`,()=>mergePlayers(right.id,left.id)));
 }
 merges.hidden=!merges.childNodes.length;
 playerState.automatic.events.forEach((event,index)=>{
  const card=document.createElement('article');card.className='card';
  const video=document.createElement('video');video.setAttribute('aria-label',`人物核对第 ${index+1} 球`);
  video.addEventListener('play',()=>document.querySelectorAll('video').forEach(v=>{if(v!==video)v.pause();}));
  if(DATA.preview?.mode==='source_segments'){const widget=segmentPlayer(video,event,`人物核对第 ${index+1} 球`);playerWidgets.push(widget);card.append(widget.element);}
  else{video.src=candidates.find(c=>c.id===event.id).clip;video.controls=true;card.append(video);}
  const body=document.createElement('div');body.className='body';const title=document.createElement('h3');title.textContent=`第 ${index+1} 球 · ${stamp(event.time_seconds)}`;
  const reason=document.createElement('p');reason.textContent=event.suggested_candidate_id&&
   playerDraft.chosen_candidates[event.id]===null&&playerDraft.assignments[event.id]
    ?'这球已手动改归属；系统原选的人物图可能不准。':event.reason;body.append(title,reason);
  const select=document.createElement('select');select.setAttribute('aria-label',`第 ${index+1} 球投篮人`);
  select.add(new Option('暂无法确定，请指定',''));select.add(new Option('暂不分配，明确跳过','__skip__'));
  for(const player of playerDraft.players)select.add(new Option(playerLabel(player),player.id));
  select.value=playerDraft.skipped.includes(event.id)?'__skip__':playerDraft.assignments[event.id]||'';
  select.addEventListener('change',()=>setGoalOwner(event.id,select.value));body.append(select);
  function chooseCandidate(candidate){
   let player=playerDraft.players.find(p=>p.id===playerDraft.assignments[event.id]);
   if(!player||Object.values(playerDraft.assignments).filter(id=>id===player.id).length>1)player=addPlayer(candidate.id);
   player.representative=candidate.id;playerDraft.assignments[event.id]=player.id;playerDraft.chosen_candidates[event.id]=candidate.id;
   playerDraft.skipped=playerDraft.skipped.filter(id=>id!==event.id);playerChanged();renderPlayers();
  }
  function candidateButton(candidate){
   const button=playerButton('',()=>chooseCandidate(candidate));
   button.setAttribute('aria-label',`第 ${index+1} 球选择人物图 ${candidate.track+1}`);
   button.classList.toggle('active',playerDraft.chosen_candidates[event.id]===candidate.id);
   if(candidate.review_context_url){
    const scene=document.createElement('div');scene.className='portrait';
    const img=document.createElement('img');img.src=candidate.review_context_url;img.alt='绿框标出的候选人和黄圈标出的球';scene.append(img);button.append(scene);
   }else button.append(playerPicture(candidate.id));
   const label=document.createElement('span');label.textContent='这是投篮人';button.append(label);return button;
  }
  const priority=event.candidates.filter(candidate=>candidate.review_priority);
  if(priority.length){
   const prompt=document.createElement('p');prompt.textContent='这球需要确认投篮人。绿框是候选人，黄圈是球。看不清时保留“暂无法确定”。';
   const choices=document.createElement('div');choices.className='people-options priority-options';
   for(const candidate of priority)choices.append(candidateButton(candidate));
   body.append(prompt,choices);
  }
  body.append(playerButton('这球单独成组',()=>{
   const representative=playerDraft.chosen_candidates[event.id];
   const player=addPlayer(representative);playerDraft.assignments[event.id]=player.id;
   playerDraft.skipped=playerDraft.skipped.filter(id=>id!==event.id);playerChanged();renderPlayers();
  }));
  const details=document.createElement('details');
  const remaining=event.candidates.filter(candidate=>!candidate.review_priority);
  const summary=document.createElement('summary');summary.textContent=`查看片段人物图并指定投篮人（${remaining.length}）`;details.append(summary);
  const people=document.createElement('div');people.className='people-options';
  for(const candidate of remaining)people.append(candidateButton(candidate));
  if(!event.candidates.length){const p=document.createElement('p');p.textContent='未得到清晰人物图。可新建人物并手动归属，或暂不分配。';people.append(p);}
  details.append(people);body.append(details);card.append(body);events.append(card);
 });
 if(!playerDirty)playerEl('player-message').textContent=`${playerState.automatic.events.length} 个进球 · ${playerPending()} 个待指定。人物建议需核对，保存后可以选人导出。`;
 checkPlayerSelection();updatePlayerExport();renderPlayerExport();
}
async function pollPlayers(){
 try{
  const response=await fetch('api/players/status',{cache:'no-store'});if(!response.ok)throw Error('无法读取人物匹配状态。');
  const result=await response.json();playerBusy=result.status==='running';update();
  playerEl('retry-players').hidden=true;
  if(playerBusy){const p=result.progress;playerEl('player-progress').textContent=p?`正在匹配投篮人：${p.done} / ${p.total} 球…`:'正在准备人物匹配…';setTimeout(pollPlayers,1000);return;}
  if(result.status==='failed'){playerEl('player-progress').textContent='匹配未完成。'+(result.error||'请重试。');playerEl('retry-players').hidden=false;return;}
  if(result.status==='completed'){
   if(playerDirty&&playerState?.analysis_id===result.analysis_id)return;
   playerState=result;playerDraft=structuredClone(result.decisions);playerDirty=false;
   playerEl('player-progress').textContent='';renderPlayers();
  }else{
   playerEl('player-progress').textContent='请先在第一步确认进球，再开始人物匹配。';
   playerEl('retry-players').hidden=false;
  }
 }catch(error){playerBusy=false;update();playerEl('player-progress').textContent=error.message;playerEl('retry-players').hidden=false;}
}
function renderPlayerExport(){
 if(!playerExportStatus)return;
 const result=playerExportStatus;
 const current=playerState&&result.analysis_id===playerState.analysis_id&&result.revision===playerDraft?.revision&&samePlayerSelection()&&!playerDirty;
 playerEl('player-montage').hidden=true;playerEl('player-record').hidden=true;
 if(!current){playerEl('player-result').textContent='已有一次人物集锦，但人物归属或进球选择已变化。请重新生成。';return;}
 playerEl('player-result').textContent=`已生成 ${result.player_name} 的 ${result.selected_count} 球集锦。暂不分配 ${result.skipped_count} 球。`;
 playerEl('player-montage').href=result.montage_url;playerEl('player-montage').hidden=false;
 playerEl('player-record').href=result.selection_url;playerEl('player-record').hidden=false;
}
async function enterPlayers(){
 showReviewScreen('players');
 if(playerState?.status==='completed'&&samePlayerSelection()){checkPlayerSelection();return;}
 playerEl('player-step').hidden=true;
 playerBusy=true;update();playerEl('player-progress').textContent='正在提交已确认进球…';
 playerEl('retry-players').hidden=true;
 try{await playerPost('start',selection());await pollPlayers();}
 catch(error){playerBusy=false;update();playerEl('player-progress').textContent=error.message;playerEl('retry-players').hidden=false;}
}
playerEl('next-players').addEventListener('click',enterPlayers);
playerEl('retry-players').addEventListener('click',enterPlayers);
playerEl('new-player').addEventListener('click',()=>{addPlayer();playerChanged();renderPlayers();});
playerEl('back-goals').addEventListener('click',()=>showReviewScreen('goals'));
playerEl('export-player').addEventListener('change',updatePlayerExport);
playerEl('save-players').addEventListener('click',async()=>{
 playerSubmitting=true;updatePlayerExport();
 try{
  playerDraft=await playerPost('save',{...playerDraft,...playerRequest()});playerDirty=false;
  playerEl('player-message').textContent='人物归属已保存。';renderPlayers();
 }catch(error){playerEl('player-message').textContent=error.message;}
 finally{playerSubmitting=false;updatePlayerExport();}
});
playerEl('export-one').addEventListener('click',async()=>{
 playerSubmitting=true;updatePlayerExport();
 exportKind='player';playerEl('player-result').textContent='正在提交导出…';
 playerEl('player-montage').hidden=true;playerEl('player-record').hidden=true;
 try{await playerPost('export',{...playerRequest(),player_id:playerEl('export-player').value});busy=true;update();poll();}
 catch(error){playerEl('player-message').textContent=error.message;}
 finally{playerSubmitting=false;updatePlayerExport();}
});
window.addEventListener('beforeunload',event=>{if(playerDirty){event.preventDefault();event.returnValue='';}});
