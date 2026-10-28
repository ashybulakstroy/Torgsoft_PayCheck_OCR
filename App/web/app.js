'use strict';
const freshStart=new URLSearchParams(location.search).get('new')==='1';
if(freshStart){try{localStorage.removeItem('receipt-last-job');}catch{}history.replaceState(null,'','/');}
const $ = id => document.getElementById(id);
const SVG_NS = 'http://www.w3.org/2000/svg';
const state = {job:null, stream:null, assets:new Map(), rows:new Map(), result:null, token:0,
  maxUpload:20*1024*1024, activePhase:null, completed:new Set(), reviewOnly:false, selectedRow:null, blob:null,
  editingRow:null, selectedCatalog:null, searchToken:0, saving:false, passSimilarity:87, totalFocused:false};
const stepKeys = ['geometry','ocr','layout','matching'];
function productLabel(count){const n=count%100;return count+' '+(n>=11&&n<=14?'товаров':count%10===1?'товар':count%10>=2&&count%10<=4?'товара':'товаров');}
function node(tag, className='', text=''){const element=document.createElement(tag);element.className=className;if(text!==undefined)element.textContent=text;return element;}
function icon(id){const svg=document.createElementNS(SVG_NS,'svg');const use=document.createElementNS(SVG_NS,'use');use.setAttribute('href','#i-'+id);svg.append(use);return svg;}
function toast(message){$('toast').textContent=message;$('toast').hidden=false;clearTimeout(toast.timer);toast.timer=setTimeout(()=>$('toast').hidden=true,5000);}
function setPill(text, kind){$('result-state').textContent=text;$('result-state').className='pill '+kind;}
function setProgress(value){const progress=Math.max(0,Math.min(100,value));$('progress-fill').style.width=progress+'%';$('progress-label').textContent=progress+'%';document.querySelector('[role=progressbar]').setAttribute('aria-valuenow',progress);}
function setPhase(phase, title){
  if(state.activePhase&&phase!==state.activePhase&&stepKeys.includes(state.activePhase))state.completed.add(state.activePhase);
  state.completed.delete(phase);state.activePhase=phase;
  if(['ocr','layout','matching'].includes(phase))startTransferFlow();
  $('process-title').textContent=title;
  const subtitles={input:'Сохраняем оригинал и учитываем ориентацию',geometry:'Контур → crop → выпрямленная таблица',ocr:'Распознавание выполняется на вашем компьютере',layout:'Визуализируем рамки и отделяем НДС',matching:'Названия сверяются с локальными справочниками',queued:'Можно загрузить следующее фото; оно встанет в очередь'};
  $('process-subtitle').textContent=subtitles[phase]||'Обработка фотографии';
  document.querySelectorAll('[data-step]').forEach(step=>{step.classList.toggle('active',step.dataset.step===phase);step.classList.toggle('complete',state.completed.has(step.dataset.step));});
  document.querySelectorAll('.image-card').forEach(card=>card.classList.remove('active-operation'));
  const target=phase==='geometry'?'original':phase==='input'?null:'rectified';
  const activeAsset=state.assets.get(target);if(activeAsset){activeAsset.card.classList.add('active-operation');drawAsset(activeAsset);}
}
function logEvent(text){
  if(!text||text===$('event-log').lastElementChild?.dataset.message)return;
  const item=node('li');item.dataset.message=text;
  item.append(node('time','log-time',new Date().toLocaleTimeString('ru-RU',{hour12:false})),node('span','',text));
  $('event-log').append(item);$('activity').hidden=false;
  $('event-count').textContent=$('event-log').children.length+' событий';
}
function polygon(svg, points, className, delay=0){
  const shape=document.createElementNS(SVG_NS,'polygon');shape.setAttribute('points',points.map(p=>p.join(',')).join(' '));shape.setAttribute('class',className);shape.style.animationDelay=delay+'s';svg.append(shape);return shape;
}
function needsMatchReview(row){
  return Boolean(row&&row.kind==='product'&&row.stage!=='ocr'&&!row.name_confirmed&&
    (row.similarity==null||row.similarity<(state.result?.pass_similarity??state.passSimilarity)));
}
function syncTotalHighlight(asset){
  let shape=asset.overlay.querySelector('.receipt-total-focus');
  if(!state.totalFocused||!['original','rectified'].includes(asset.key)){shape?.remove();return;}
  const color=getComputedStyle($('receipt-total')).color;
  if(!shape)shape=polygon(asset.overlay,[[0,0],[asset.width,0],[asset.width,asset.height],[0,asset.height]],'receipt-total-focus');
  shape.style.fill=color;
  shape.style.stroke=color;
}
function focusReceiptTotal(){
  focusRow(null,true);state.totalFocused=true;
  state.assets.forEach(syncTotalHighlight);
}
function drawAsset(asset){
  const svg=asset.overlay;svg.replaceChildren();svg.setAttribute('viewBox',`0 0 ${asset.width} ${asset.height}`);svg.setAttribute('preserveAspectRatio','xMidYMid meet');
  if(asset.contour)polygon(svg,asset.contour,'boundary');
  if(asset.region)polygon(svg,asset.region,'region');
  if(asset.cells?.length){
    asset.cells.filter(cell=>cell.kind!=='total').forEach((cell,index)=>{
      const shape=polygon(svg,cell.box,'cell-box'+(cell.kind==='tax'?' tax':'')+(cell.row===state.selectedRow?' focused':''),(index%30)*.025);
      shape.dataset.row=cell.row;
      if(asset.key==='rectified'&&cell.role==='name'&&needsMatchReview(state.rows.get(cell.row))){
        const underline=document.createElementNS(SVG_NS,'line');
        underline.setAttribute('x1',cell.box[3][0]+3);underline.setAttribute('y1',cell.box[3][1]-3);
        underline.setAttribute('x2',cell.box[2][0]-3);underline.setAttribute('y2',cell.box[2][1]-3);
        underline.setAttribute('class','match-warning');underline.dataset.row=cell.row;svg.append(underline);
      }
    });
  }else if(asset.boxes){asset.boxes.forEach((box,index)=>polygon(svg,box,'ocr-box',(index%30)*.025));}
  if(asset.key==='rectified'&&['ocr','layout'].includes(state.activePhase)){
    const scan=document.createElementNS(SVG_NS,'line');scan.setAttribute('x1','0');scan.setAttribute('x2',asset.width);scan.setAttribute('y1','0');scan.setAttribute('y2','0');scan.setAttribute('class','scan-line');svg.append(scan);
  }
  syncTotalHighlight(asset);
}
function openImage(key){const asset=state.assets.get(key);if(!asset)return;$('dialog-title').textContent=asset.title;$('dialog-image').replaceChildren(asset.stage.cloneNode(true));$('image-dialog').showModal();}
function followProcessing(){
  cancelAnimationFrame(followProcessing.frame);
  followProcessing.frame=requestAnimationFrame(()=>{
    if(state.result||!state.activePhase)return;
    const asset=state.assets.get('rectified')||state.assets.get('crop')||state.assets.get('original');
    const targets=[[$('input-scroll'),asset?.stage],[$('output-scroll'),$('receipt-body').lastElementChild]];
    for(const [panel,target] of targets){
      if(!target||!target.getClientRects().length)continue;
      const bounds=target.getBoundingClientRect(),viewport=panel.getBoundingClientRect();
      panel.scrollTo({top:panel.scrollTop+bounds.bottom-viewport.bottom+16,behavior:'instant'});
    }
  });
}
function animateRowTransfer(row){
  if(row.kind==='total'||matchMedia('(prefers-reduced-motion: reduce)').matches)return;
  const token=state.token;
  // Measure after the processing scroll has positioned both panels.
  requestAnimationFrame(()=>requestAnimationFrame(()=>{
    if(token!==state.token||state.result)return;
    const asset=state.assets.get('rectified');
    const target=[...$('receipt-body').children].find(element=>Number(element.dataset.row)===row.row)?.children[1];
    if(!asset||!target)return;
    const source=[...asset.overlay.querySelectorAll('.cell-box')].find(shape=>Number(shape.dataset.row)===row.row);
    const from=(source||asset.stage).getBoundingClientRect(),to=target.getBoundingClientRect();
    const left=$('input-scroll').getBoundingClientRect(),right=$('output-scroll').getBoundingClientRect();
    if(!target.getClientRects().length||left.right>=right.right)return;
    const clamp=(value,min,max)=>Math.max(min,Math.min(max,value));
    const width=Math.min(260,left.width*.65,right.width*.65);
    const x=clamp(from.left+from.width/2-width/2,left.left+12,left.right-width-12);
    const y=clamp(from.top+from.height/2-24,left.top+12,left.bottom-60);
    const endX=clamp(to.left,to.left,right.right-width-12),endY=clamp(to.top,right.top+12,right.bottom-60);
    document.querySelectorAll('.receipt-transfer').forEach((item,index,items)=>{if(index<items.length-3)item.remove();});
    const card=node('div','receipt-transfer');card.setAttribute('aria-hidden','true');
    card.append(node('strong','',row.name??row.name_raw??'Строка чека'),node('span','',`${row.quantity??row.quantity_raw??'—'} × ${displayPrice(row.price??row.price_raw)}`));
    Object.assign(card.style,{left:x+'px',top:y+'px',width:width+'px'});document.body.append(card);
    const dx=endX-x,dy=endY-y;
    const animation=card.animate([
      {transform:'translate(0,0) scale(.94)',opacity:0},
      {transform:`translate(${dx*.15}px,${dy*.15-18}px) scale(1)`,opacity:1,offset:.2},
      {transform:`translate(${dx}px,${dy}px) scale(.96)`,opacity:1,offset:.85},
      {transform:`translate(${dx}px,${dy}px) scale(1)`,opacity:0}
    ],{duration:750,easing:'ease-in-out'});
    animation.finished.catch(()=>{}).finally(()=>card.remove());
  }));
}
function stopTransferFlow(){
  clearInterval(startTransferFlow.timer);startTransferFlow.timer=null;
  document.querySelectorAll('.receipt-flow-dot,.receipt-transfer').forEach(element=>element.remove());
}
function startTransferFlow(){
  if(startTransferFlow.timer||matchMedia('(prefers-reduced-motion: reduce)').matches)return;
  const pulse=()=>{
    if(state.result)return;
    const asset=state.assets.get('rectified');if(!asset)return;
    const left=$('input-scroll').getBoundingClientRect(),right=$('output-scroll').getBoundingClientRect();
    if(left.right>=right.right)return;
    const source=asset.stage.getBoundingClientRect();
    const last=$('receipt-body').lastElementChild;
    const target=(last||$('receipt-area')).getBoundingClientRect();
    const clamp=(value,min,max)=>Math.max(min,Math.min(max,value));
    const x=clamp(source.right-24,left.left+20,left.right-20);
    const y=clamp(source.bottom-45,left.top+30,left.bottom-30);
    const endX=right.left+Math.min(85,right.width*.18);
    const endY=clamp(target.bottom-25,right.top+50,right.bottom-35);
    const dot=node('div','receipt-flow-dot');dot.setAttribute('aria-hidden','true');
    Object.assign(dot.style,{left:x+'px',top:y+'px'});document.body.append(dot);
    const animation=dot.animate([
      {transform:'translate(0,0) scale(.6)',opacity:0},
      {transform:`translate(${(endX-x)*.2}px,${(endY-y)*.2-20}px) scale(1)`,opacity:1,offset:.2},
      {transform:`translate(${endX-x}px,${endY-y}px) scale(.5)`,opacity:0}
    ],{duration:1100,easing:'ease-in-out'});
    animation.finished.catch(()=>{}).finally(()=>dot.remove());
  };
  pulse();startTransferFlow.timer=setInterval(pulse,420);
}
function addArtifact(data){
  let asset=state.assets.get(data.key);
  if(asset){Object.assign(asset,data);asset.img.src=data.url;asset.heading.textContent=data.title;asset.subtitle.textContent=data.subtitle||'';asset.dimensions.textContent=`${data.width} × ${data.height} px`;drawAsset(asset);return;}
  const card=node('article','image-card'+(['header','crop'].includes(data.key)?' debug-only':''));card.dataset.key=data.key;
  const header=node('div','image-card-header');const headings=node('div');const heading=node('h3','',data.title);const subtitle=node('p','',data.subtitle||'');headings.append(heading,subtitle);
  header.append(headings,node('span','operation-tag',data.role==='original'?'SOURCE':data.key.toUpperCase()));
  const stage=node('div','image-stage');const img=document.createElement('img');img.addEventListener('load',followProcessing);img.src=data.url;img.alt=data.title;img.loading='eager';
  const overlay=document.createElementNS(SVG_NS,'svg');overlay.setAttribute('aria-hidden','true');stage.append(img,overlay);
  stage.addEventListener('click',()=>openImage(data.key));
  const tools=node('div','image-tools');const dimensions=node('span','',`${data.width} × ${data.height} px`);const expand=node('button','','Рассмотреть');expand.type='button';expand.append(icon('expand'));expand.addEventListener('click',()=>openImage(data.key));tools.append(dimensions,expand);
  card.append(header,stage,tools);$('image-list').append(card);
  const order=['original','header','crop','rectified'];
  [...$('image-list').children].sort((a,b)=>order.indexOf(a.dataset.key)-order.indexOf(b.dataset.key)).forEach(item=>$('image-list').append(item));
  asset={...data,card,heading,subtitle,stage,img,overlay,dimensions};state.assets.set(data.key,asset);drawAsset(asset);
  followProcessing();
}
function focusRow(rowId, persist=false){
  state.totalFocused=false;state.assets.forEach(syncTotalHighlight);
  state.selectedRow=rowId;
  const updateFocus=()=>{const asset=state.assets.get('rectified');if(asset){asset.overlay.classList.add('static-focus');asset.overlay.querySelectorAll('.cell-box').forEach(shape=>shape.classList.toggle('focused',state.selectedRow!==null&&Number(shape.dataset.row)===state.selectedRow));}};
  updateFocus();
  $('receipt-body').querySelectorAll('tr').forEach(row=>row.classList.toggle('selected',Number(row.dataset.row)===rowId));
  clearTimeout(focusRow.timer);
  if(!persist)focusRow.timer=setTimeout(()=>{if(state.selectedRow===rowId){state.selectedRow=null;updateFocus();}},900);
}
function formatPercent(value){return value===null||value===undefined?'—':Number(value).toFixed(1)+'%';}
function formatMoney(value){if(value===null||value===undefined||value==='')return '—';const [whole,decimal='00']=String(value).split('.');return whole.replace(/\B(?=(\d{3})+(?!\d))/g,' ') + ',' + decimal.padEnd(2,'0')+' ₸';}
function displayPrice(value){return String(value??'').replace(/\s*(?:KZT|ТГ|ТЕНГЕ|[TТ₸])\s*$/iu,'').trim()||'—';}
function isReceiptNumber(value){
  if(value==null||typeof value==='boolean'||typeof value==='object')return false;
  let text=String(value).replace(/\s+/g,'').replace(/(?:KZT|ТГ|ТЕНГЕ|[TТ₸])$/iu,'');
  if(text.length>40||!/^\d+(?:[.,]\d+)*$/.test(text))return false;
  if(text.includes('.')&&text.includes(','))text=text.replace(text.lastIndexOf('.')>text.lastIndexOf(',')?/,/g:/\./g,'');
  text=text.replace(/,/g,'.');
  return (text.match(/\./g)||[]).length<=1&&Number.isFinite(Number(text))&&Number(text)>=0&&Number(text)<1e15;
}
function numericReceiptCell(row,field){
  const value=row[field]??row[field+'_raw'];
  const invalid=Array.isArray(row.invalid_fields)?row.invalid_fields.includes(field):!isReceiptNumber(value);
  const shown=field==='price'?displayPrice(value):(value==null||String(value).trim()===''?'—':String(value));
  const cell=node('td',invalid?'invalid-number':'',shown);cell.dataset.field=field;
  if(invalid){cell.setAttribute('aria-invalid','true');cell.title=field==='quantity'?'Количество не распознано числом. Нажмите для исправления.':'Цена не распознана числом. Нажмите для исправления.';}
  return cell;
}
function makeReceiptRow(row,ordinal){
  const draft=row.stage==='ocr';
  const lowMatch=needsMatchReview(row);
  const tr=node('tr',(lowMatch?'needs-check ':row.kind!=='product'?'special-row ':'')+(draft?'ocr-draft':''));tr.dataset.row=row.row;
  if(row.kind==='tax'&&row.vat_status)tr.classList.add(row.vat_status==='match'?'vat-passed':'vat-failed');
  tr.addEventListener('mouseenter',()=>focusRow(row.row,true));tr.addEventListener('click',()=>{focusRow(row.row,true);if(state.result)editRow(row.row);state.assets.get('rectified')?.card.scrollIntoView({behavior:'smooth',block:'nearest'});});
  const printedNumber=String(row.item_number_raw??'').trim();
  const hasPrintedNumber=/^\d+$/.test(printedNumber)&&Number(printedNumber)>0;
  const numberCell=node('td',hasPrintedNumber?'':'inferred-number',hasPrintedNumber?printedNumber:String(ordinal));
  if(!hasPrintedNumber)numberCell.title='Номер не распознан: указан порядковый номер строки таблицы.';
  tr.append(numberCell);
  const description=node('td');const kindNames={tax:'НДС',service:row.name_raw||'Доставка',total:'Итог',unreadable:'Не удалось прочитать название'};
  const name=row.name??row.name_raw;
  description.append(node('div','item-name',row.kind==='tax'&& !/[нН][дД][сС]/.test(name)?'НДС':name||kindNames[row.kind]||'—'));
  if(row.kind==='product'){
    const debug=node('div','receipt-debug debug-only');
    debug.append(node('span','item-reference','OCR: '+row.name_raw));
    if(row.match?.best)debug.append(node('span','item-reference','Справочник: '+row.match.best.name));
    if(row.article)debug.append(node('span','item-reference','Артикул '+row.article));
    if(row.quantity_source==='cell_ocr')debug.append(node('span','item-reference',`Количество из отдельной ячейки: ${row.quantity} · OCR ${formatPercent((row.quantity_ocr?.views?.[0]?.confidence??0)*100)}`));
    const metrics=node('div','item-metrics');metrics.append(node('span','metric','Уверенность OCR '+formatPercent(row.ocr_name_confidence_pct)));
    if(row.match?.source==='manual_mapping'){
      metrics.append(node('span','metric pass','База ручных сопоставлений'));
      metrics.append(node('span','metric','Автоматическое сходство '+formatPercent(row.match.automatic_match?.best?.score)));
    }else if(row.similarity!=null)metrics.append(node('span','metric '+(needsMatchReview(row)?'low':'pass'),'Сходство '+formatPercent(row.similarity)));
    else if(draft)metrics.append(node('span','metric','Распознавание…'));
    debug.append(metrics);
    if(row.review_reason)debug.append(node('div','item-reason',row.review_reason));
    description.append(debug);
  }
  tr.append(description,numericReceiptCell(row,'quantity'),numericReceiptCell(row,'price'));
  return tr;
}
function renderRows(){
  const rows=[...state.rows.values()].filter(row=>row.kind!=='total').sort((a,b)=>a.row-b.row);const products=rows.filter(row=>row.kind==='product');const review=rows.filter(row=>row.needs_review);
  $('product-count').textContent=productLabel(products.length);$('review-count').textContent=review.length;
  const selected=state.reviewOnly?rows.filter(row=>row.needs_review):rows;
  const ordinals=new Map(rows.map((row,index)=>[row.row,index+1]));
  $('receipt-body').replaceChildren(...selected.map(row=>makeReceiptRow(row,ordinals.get(row.row))));$('receipt-empty-filter').hidden=selected.length!==0;
  $('receipt-area').hidden=false;$('review-note').hidden=!review.length;
  const asset=state.assets.get('rectified');if(asset)drawAsset(asset);
}
function showMathSuccess(){
  document.querySelector('.math-success-flash')?.remove();
  const panel=$('output-scroll').getBoundingClientRect();
  const top=Math.max(0,panel.top),bottom=Math.min(innerHeight,panel.bottom);
  if(bottom<=top)return;
  const badge=node('div','math-success-flash');badge.setAttribute('role','status');badge.setAttribute('aria-label','Сумма и НДС проверены: всё сходится');
  const svg=document.createElementNS(SVG_NS,'svg');svg.setAttribute('viewBox','0 0 100 100');svg.setAttribute('aria-hidden','true');
  const check=document.createElementNS(SVG_NS,'path');check.setAttribute('d','M22 51 L42 71 L79 30');svg.append(check);badge.append(svg);
  Object.assign(badge.style,{left:(panel.left+panel.width/2)+'px',top:((top+bottom)/2)+'px'});document.body.append(badge);
  clearTimeout(showMathSuccess.timer);showMathSuccess.timer=setTimeout(()=>badge.remove(),1300);
}
function renderMath(){
  const result=state.result, calc=result?.calculation;if(!calc){$('math-panel').hidden=true;return;}
  const status=calc.validation_status??calc.status;
  if(status==='match'&&renderMath.previousStatus!=='match')showMathSuccess();
  renderMath.previousStatus=status;
  $('math-panel').hidden=true;$('math-panel').dataset.status=status;
  document.querySelector('.receipt-total').classList.toggle('math-passed',status==='match');
  document.querySelector('.receipt-total').classList.toggle('math-failed',status!=='match');
  const reasons={match:'Сумма и НДС проверены',mismatch:'Сумма не совпадает с фотографией',incomplete:'Не все количества или цены прочитаны',missing_original_total:'Итог с фотографии не прочитан',vat_mismatch:calc.vat?.reason||'НДС не прошёл проверку'};
  document.querySelector('.receipt-total').title=reasons[status]||'Требуется проверка';
  if(state.totalFocused)state.assets.forEach(syncTotalHighlight);
  const labels={match:'Суммы совпадают',mismatch:'Суммы не совпадают',incomplete:'Не все числа прочитаны',missing_original_total:'Укажите итог с фото',vat_mismatch:'Проверьте НДС'};
  $('math-status').textContent=labels[status]||'Проверка';$('math-status').className='pill '+(status==='match'?'done':'error');
  $('calculated-total').textContent=formatMoney(calc.computed_total);$('expected-total-label').textContent=formatMoney(calc.expected_total);$('total-difference').textContent=formatMoney(calc.difference);
  $('math-breakdown').textContent=`Товары: ${formatMoney(calc.products_subtotal)} · Доставка: ${formatMoney(calc.services_subtotal)} · НДС: ${formatMoney(calc.tax_subtotal)}`;
  $('math-issues').replaceChildren();
  (calc.missing_rows||[]).forEach(issue=>{const button=node('button','math-issue',`${issue.name||'Строка '+issue.row}: проверьте ${issue.fields.map(f=>f==='quantity'?'количество':'цену').join(' и ')}`);button.type='button';button.addEventListener('click',()=>editRow(issue.row));$('math-issues').append(button);});
  if(calc.status==='mismatch')$('math-issues').append(node('p','math-explanation','Сверьте количество и цену строк с фотографией. Нажмите на строку чека для редактирования.'));
  $('finalize-receipt').disabled=!result.can_finalize||result.finalized||state.saving;
  $('finalize-receipt').textContent=result.finalized?'Чек проверен':'Завершить проверку';
  $('finalize-hint').textContent=result.finalized?'Названия утверждены, сумма совпадает.':result.review_count?`Осталось проверить строк: ${result.review_count}`:calc.status==='match'?'Можно завершить проверку.':'Исправьте числа или итог с фотографии.';
}
function showCandidates(items){
  $('catalog-options').replaceChildren();
  if(!items.length){$('catalog-options').append(node('p','editor-evidence','Точный товар не найден. Уточните запрос или отредактируйте и утвердите название вручную.'));return;}
  items.forEach(item=>{const button=node('button','catalog-option');button.type='button';button.append(node('strong','',item.name),node('span','',item.articles.length?'Артикул '+item.articles.join(' / '):'Артикул не указан'));button.addEventListener('click',()=>selectCandidate(item));$('catalog-options').append(button);});
}
function selectCandidate(item, explicit=true){
  state.catalogChosen=explicit;
  state.selectedCatalog=item;$('edit-name').value=item.name;$('article-field').hidden=item.articles.length<=1;$('edit-article').replaceChildren();
  if(item.articles.length>1)$('edit-article').append(new Option('Выберите артикул',''));
  item.articles.forEach(article=>$('edit-article').append(new Option(article,article)));
  const row=state.rows.get(state.editingRow);if(row?.article&&item.articles.includes(row.article))$('edit-article').value=row.article;
  $('editor-hint').textContent='Выбран товар из справочника';
}
function editRow(rowId, scroll=true, highlight=true){
  if(!state.result)return;const row=state.rows.get(Number(rowId));if(!row)return;
  state.editingRow=row.row;state.selectedCatalog=null;state.catalogChosen=false;state.searchToken++;
  $('editor-panel').hidden=false;$('editor-panel').open=true;$('edit-row').value=String(row.row);
  $('edit-name').value=row.name??row.name_raw;$('edit-quantity').value=row.quantity??row.quantity_raw;$('edit-price').value=row.price??row.price_raw;
  $('edit-ocr-evidence').textContent='Исходный OCR: '+row.name_raw;
  $('catalog-search').value='';$('catalog-picker').hidden=row.kind!=='product';$('article-field').hidden=true;$('editor-error').hidden=true;
  $('editor-hint').textContent=row.needs_review?'Требует проверки':'Можно исправить данные';
  const options=(row.match?.top_matches||[]).map(item=>({key:item.normalized,name:item.name,articles:[...new Set(item.sources.map(s=>s.article).filter(Boolean))]}));showCandidates(options);
  $('accept-match').hidden=!row.match?.best||row.kind!=='product';
  if(row.catalog_key){const current=options.find(item=>item.key===row.catalog_key)||{key:row.catalog_key,name:row.name,articles:row.articles||[]};selectCandidate(current,false);}
  if(scroll)$('editor-panel').scrollIntoView({behavior:'smooth',block:'nearest'});
  if(highlight)focusRow(row.row,true);
}
function setupEditor(){
  $('editor-panel').hidden=false;const selected=state.editingRow;
  const editable=[...state.rows.values()].filter(row=>row.kind!=='total').sort((a,b)=>a.row-b.row);
  $('edit-row').replaceChildren(...editable.map((row,index)=>new Option(`${index+1}. ${(row.name??row.name_raw)||'Не прочитано'}`,String(row.row))));
  $('edit-expected-total').value=state.result.calculation.expected_total??'';
  const row=editable.find(row=>row.row===selected)||editable.find(row=>row.needs_review)||editable[0];
  const wasOpen=$('editor-panel').open;
  $('editor-row-fields').hidden=!row;if(row)editRow(row.row,false,false);
  $('editor-panel').open=wasOpen||Boolean(row?.needs_review);
}
async function sendEdit(payload){
  if(!state.result||state.saving)return;
  state.saving=true;$('save-row').disabled=true;$('save-total').disabled=true;$('editor-error').hidden=true;renderMath();
  try{
    const response=await fetch(`/api/jobs/${state.job}/edits`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({revision:state.result.revision,...payload})});
    const data=await response.json();if(!response.ok)throw Error(data.error||'Не удалось сохранить изменения');
    finish(data.result);toast(data.mapping_saved?'Выбор сохранён в базе ручных сопоставлений':data.result.finalized?'Чек проверен. Суммы совпадают.':'Изменения сохранены, сумма пересчитана');
  }catch(error){$('editor-error').textContent=error.message;$('editor-error').hidden=false;toast(error.message);}
  finally{state.saving=false;$('save-row').disabled=false;$('save-total').disabled=false;renderMath();}
}
async function saveEditedRow(){
  const row=state.rows.get(state.editingRow);if(!row)return;
  const payload={row:row.row,name:$('edit-name').value,confirm_name:true};
  if(state.selectedCatalog&&$('edit-name').value===state.selectedCatalog.name){payload.catalog_key=state.selectedCatalog.key;payload.learn_mapping=Boolean(state.catalogChosen);payload.article=state.selectedCatalog.articles.length>1?$('edit-article').value:state.selectedCatalog.articles[0]||'';}
  for(const [field,id] of [['quantity','edit-quantity'],['price','edit-price']])if($(id).value!==(row[field]??row[field+'_raw']))payload[field]=$(id).value;
  await sendEdit(payload);
}
function reset(){
  window.OcrActivity?.finish(false);
  renderMath.previousStatus=null;clearTimeout(showMathSuccess.timer);document.querySelector('.math-success-flash')?.remove();
  stopTransferFlow();
  document.querySelectorAll('.receipt-transfer').forEach(card=>card.remove());
  state.token++;state.stream?.close();state.stream=null;state.job=null;state.assets.clear();state.rows.clear();state.result=null;state.completed.clear();state.activePhase=null;state.selectedRow=null;state.reviewOnly=false;
  if(state.blob){URL.revokeObjectURL(state.blob);state.blob=null;}
  state.totalFocused=false;state.editingRow=null;state.selectedCatalog=null;state.searchToken++;$('editor-panel').hidden=true;$('editor-panel').open=false;$('math-panel').hidden=true;
  document.querySelector('.receipt-total').classList.remove('math-passed','math-failed');document.querySelector('.receipt-total').removeAttribute('title');
  $('image-list').replaceChildren();$('receipt-body').replaceChildren();$('event-log').replaceChildren();$('receipt-area').hidden=true;$('empty-result').hidden=true;$('error-card').hidden=true;$('geometry-note').hidden=true;$('exports').hidden=true;$('activity').hidden=true;$('raw-text').textContent='';$('order-number').textContent='';$('receipt-date').textContent='Данные с фотографии';$('receipt-total').textContent='—';$('receipt-meta').textContent='Распознавание выполняется…';$('review-toggle').setAttribute('aria-pressed','false');$('review-note').hidden=true;$('process-card').classList.add('busy');
  document.querySelectorAll('[data-step]').forEach(step=>step.classList.remove('active','complete'));setProgress(0);setPill('Обработка','processing');
}
function showError(message){
  window.OcrActivity?.finish(false);
  stopTransferFlow();
  state.stream?.close();$('process-card').classList.remove('busy');$('error-card').hidden=false;$('error-text').textContent=message;$('process-title').textContent='Обработка остановлена';$('process-subtitle').textContent='Исходное фото сохранено';setPill('Нужна проверка','error');logEvent(message);
  document.querySelectorAll('.image-card').forEach(card=>card.classList.remove('active-operation'));
}
function finish(result){
  if(!result)return;
  window.OcrActivity?.finish(true);
  stopTransferFlow();
  if(!state.result)focusRow(null,true);
  state.result=result;state.stream?.close();state.stream=null;state.rows=new Map(result.rows.map(row=>[row.row,row]));renderRows();
  $('empty-result').hidden=true;$('error-card').hidden=true;$('process-card').classList.remove('busy');setProgress(100);
  $('process-title').textContent=result.finalized?'Чек проверен и готов':'Распознавание завершено';$('process-subtitle').textContent=`${productLabel(result.product_count)} · ${(result.elapsed_ms/1000).toFixed(1)} с · ${result.catalog_names.toLocaleString('ru-RU')} названий в справочнике`;
  setPill(result.finalized?'Чек проверен':result.can_finalize?'Можно утвердить':'Нужна проверка',result.finalized?'done':result.can_finalize?'done':'processing');
  document.querySelectorAll('[data-step]').forEach(step=>{step.classList.remove('active');step.classList.add('complete');});
  document.querySelectorAll('.image-card').forEach(card=>card.classList.remove('active-operation'));
  $('order-number').textContent=result.order_number?'№ '+result.order_number:'';$('receipt-date').textContent=result.date||result.filename;
  $('receipt-total').textContent=result.calculation?formatMoney(result.calculation.computed_total):result.total_raw||'Не прочитан';$('receipt-meta').textContent=result.finalized?'Проверено пользователем · итог совпадает с фотографией':'Черновик · исходный OCR сохранён отдельно';
  renderMath();setupEditor();
  $('download-json').href=result.downloads.json;$('download-csv').href=result.downloads.csv;$('exports').hidden=false;$('raw-text').textContent=result.raw_text;
  if(result.warnings?.length){$('geometry-note').hidden=false;$('geometry-note').textContent=result.warnings.some(w=>w.includes('не удалось'))?result.warnings.join(' · '):'Граница таблицы содержит неоднозначные участки. Проверьте результат выпрямления по исходному фото.';}
}
function consume(event){
  if(event.progress!==undefined)setProgress(event.progress);
  if(event.type==='phase'){setPhase(event.phase==='layout'&&event.progress<50?'geometry':event.phase,event.title);window.OcrActivity?.phase(event.phase,event.title);logEvent(event.title);}
  else if(event.type==='artifact'){addArtifact(event.artifact);logEvent(event.artifact.title);}
  else if(event.type==='contour'){const asset=state.assets.get(event.target);if(asset){asset.contour=event.polygon;drawAsset(asset);}logEvent(event.title);}
  else if(event.type==='region'){const asset=state.assets.get(event.target);if(asset){asset.region=event.box;drawAsset(asset);}}
  else if(event.type==='boxes'){const asset=state.assets.get(event.target);if(asset){asset.boxes=event.boxes;asset.cells=null;drawAsset(asset);}logEvent(`${event.title}: ${event.boxes.length}`);}
  else if(event.type==='cells'){const asset=state.assets.get(event.target);if(asset){asset.cells=event.cells;drawAsset(asset);}logEvent(`${event.title} · ${event.cells.length} ячеек`);}
  else if(event.type==='header'){$('receipt-area').hidden=false;$('order-number').textContent=event.order_number?'№ '+event.order_number:'';$('receipt-date').textContent=event.date||'Дата не прочитана';}
  else if(event.type==='draft_rows'){window.OcrActivity?.rows(event.rows);if(event.pass_similarity!==undefined)state.passSimilarity=event.pass_similarity;state.rows=new Map(event.rows.map(row=>[row.row,row]));renderRows();$('receipt-meta').textContent='Черновик из OCR · названия ещё не сверены';}
  else if(event.type==='row'){state.rows.set(event.row.row,event.row);renderRows();focusRow(event.row.row);logEvent(event.title);}
  else if(event.type==='message'){window.OcrActivity?.message(event.title);$('process-subtitle').textContent=event.title;logEvent(event.title);}
  else if(event.type==='ui_status'){window.OcrActivity?.uiStatus(event);$('process-subtitle').textContent=event.title;if(event.rows_total||event.matched_products!==undefined)logEvent(window.OcrActivity ? document.getElementById('ocr-activity-details').textContent : event.title);}
  else if(event.type==='complete'){finish(event.result);logEvent('Обработка завершена');}
  else if(event.type==='error')showError(event.title);
  if(event.type!=='complete'&&event.type!=='error')followProcessing();
  if(event.type==='row')animateRowTransfer(event.row);
}
function listen(jobId, token){
  state.stream=new EventSource(`/api/jobs/${jobId}/events`);
  state.stream.onopen=()=>{if(token===state.token)window.OcrActivity?.heartbeat();};
  state.stream.addEventListener('heartbeat',()=>{if(token===state.token)window.OcrActivity?.heartbeat();});
  state.stream.addEventListener('pipeline',event=>{if(token!==state.token)return;window.OcrActivity?.heartbeat();try{consume(JSON.parse(event.data));}catch(error){console.error(error);toast('Не удалось отобразить обновление. Обновите страницу.');}});
  state.stream.onerror=async()=>{
    if(token!==state.token)return;
    window.OcrActivity?.connectionLost();
    try{const response=await fetch(`/api/jobs/${jobId}`);if(!response.ok)throw Error();const job=await response.json();if(job.status==='done')finish(job.result);else if(job.status==='failed')showError(job.error);else $('process-subtitle').textContent='Восстанавливаем связь с обработкой…';}catch{ $('process-subtitle').textContent='Нет связи с сервером. Ожидаем подключения…'; }
  };
}
async function upload(file){
  if(!file)return;
  if(file.size>state.maxUpload){toast(`Максимальный размер фото — ${Math.floor(state.maxUpload/1024/1024)} МБ`);return;}
  if(file.type&&!['image/jpeg','image/png','image/webp'].includes(file.type)){toast('Выберите фотографию JPG, PNG или WEBP');return;}
  reset();window.OcrActivity?.start();const token=state.token;$('drop-zone').hidden=true;$('input-note').hidden=false;state.blob=URL.createObjectURL(file);
  logEvent('Начало обработки: '+file.name);
  const preview=new Image();preview.src=state.blob;preview.onload=()=>{if(token===state.token&&!state.assets.has('original'))addArtifact({key:'original',role:'original',title:'Исходное фото',subtitle:file.name,url:preview.src,width:preview.naturalWidth,height:preview.naturalHeight});};
  $('process-title').textContent='Загружаем фотографию';$('process-subtitle').textContent=file.name;
  try{
    const response=await fetch('/api/jobs?filename='+encodeURIComponent(file.name||'clipboard.png'),{method:'POST',headers:{'Content-Type':file.type||'application/octet-stream'},body:file});
    const job=await response.json();if(!response.ok)throw Error(job.error||'Не удалось загрузить фотографию');if(token!==state.token)return;
    state.job=job.id;try{localStorage.setItem('receipt-last-job',job.id);}catch{}listen(job.id,token);
  }catch(error){if(token===state.token)showError(error.message||'Не удалось связаться с сервером');}
}
async function restore(){
  if(freshStart)return;
  let id;try{id=localStorage.getItem('receipt-last-job');}catch{return;}if(!id||!/^\w{32}$/.test(id))return;
  const token=state.token;
  try{
    const response=await fetch(`/api/jobs/${id}`);if(!response.ok)return;const job=await response.json();if(token!==state.token)return;
    state.job=id;$('drop-zone').hidden=true;$('empty-result').hidden=true;
    job.artifacts.forEach(addArtifact);setProgress(job.progress);
    if(job.status==='done'){
      finish(job.result);
      logEvent('Восстановлен сохранённый результат: '+job.filename);
      const layoutResponse=await fetch(`/api/jobs/${id}/files/layout.json`);
      if(layoutResponse.ok&&token===state.token){const data=await layoutResponse.json();const asset=state.assets.get('rectified');if(asset){asset.cells=data.cells;drawAsset(asset);}}
    }else if(job.status==='failed')showError(job.error);else{window.OcrActivity?.start();$('process-card').classList.add('busy');setPill('Обработка','processing');listen(id,token);}
  }catch{}
}
document.querySelector('.app-header').addEventListener('click',event=>{
  if(event.target.closest('.debug-toggle'))return;
  if(event.button!==0||event.ctrlKey||event.metaKey||event.shiftKey||event.altKey)return;
  event.preventDefault();try{localStorage.removeItem('receipt-last-job');}catch{}location.assign('/?new=1');
});
$('debug-toggle').checked=false;
$('debug-toggle').addEventListener('change',event=>document.body.classList.toggle('debug-mode',event.target.checked));
['choose-main','choose-top','retry-button'].forEach(id=>$(id).addEventListener('click',event=>{event.stopPropagation();$('file-input').click();}));
$('drop-zone').addEventListener('click',event=>{if(!event.target.closest('button'))$('file-input').click();});
$('drop-zone').addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();$('file-input').click();}});
$('file-input').addEventListener('change',event=>{upload(event.target.files[0]);event.target.value='';});
document.addEventListener('paste',event=>{const item=[...(event.clipboardData?.items||[])].find(item=>item.type.startsWith('image/'));if(item){event.preventDefault();upload(item.getAsFile());}});
let dragDepth=0;
document.addEventListener('dragenter',event=>{if(event.dataTransfer?.types.includes('Files')){event.preventDefault();dragDepth++;$('drop-overlay').hidden=false;}});
document.addEventListener('dragover',event=>{if(event.dataTransfer?.types.includes('Files')){event.preventDefault();event.dataTransfer.dropEffect='copy';}});
document.addEventListener('dragleave',()=>{dragDepth=Math.max(0,dragDepth-1);if(!dragDepth)$('drop-overlay').hidden=true;});
document.addEventListener('drop',event=>{event.preventDefault();dragDepth=0;$('drop-overlay').hidden=true;if(event.dataTransfer?.files.length)upload(event.dataTransfer.files[0]);});
$('review-toggle').addEventListener('click',()=>{state.reviewOnly=!state.reviewOnly;$('review-toggle').setAttribute('aria-pressed',state.reviewOnly);renderRows();});
$('close-dialog').addEventListener('click',()=>$('image-dialog').close());
$('image-dialog').addEventListener('click',event=>{if(event.target===$('image-dialog'))$('image-dialog').close();});
$('print-button').addEventListener('click',()=>window.print());
document.querySelector('.receipt-total').addEventListener('mouseenter',focusReceiptTotal);
$('row-editor').addEventListener('submit',event=>{event.preventDefault();saveEditedRow();});
$('edit-row').addEventListener('change',()=>editRow(Number($('edit-row').value),false));
$('edit-name').addEventListener('input',()=>{if(state.selectedCatalog&&$('edit-name').value!==state.selectedCatalog.name){state.selectedCatalog=null;$('article-field').hidden=true;}});
$('catalog-search').addEventListener('input',()=>{
  clearTimeout(state.searchTimer);const query=$('catalog-search').value;const token=++state.searchToken;
  state.searchTimer=setTimeout(async()=>{if(query.trim().length<2)return;try{const response=await fetch('/api/catalog?q='+encodeURIComponent(query));const data=await response.json();if(token===state.searchToken)showCandidates(data.items||[]);}catch{toast('Не удалось открыть справочник');}},250);
});
$('accept-match').addEventListener('click',()=>{const row=state.rows.get(state.editingRow);const best=row?.match?.best;if(!best)return;selectCandidate({key:best.normalized,name:best.name,articles:[...new Set(best.sources.map(source=>source.article).filter(Boolean))]});if(state.selectedCatalog.articles.length<=1)saveEditedRow();});
$('save-total').addEventListener('click',()=>sendEdit({expected_total:$('edit-expected-total').value}));
$('finalize-receipt').addEventListener('click',()=>sendEdit({finalize:true}));
fetch('/api/health').then(response=>{if(!response.ok)throw Error();return response.json();}).then(config=>{state.maxUpload=config.max_upload_bytes;$('upload-limit').textContent=Math.floor(state.maxUpload/1024/1024);$('connection-text').textContent='Сервер доступен';restore();}).catch(()=>{$('connection-dot').style.background='#dba55a';$('connection-text').textContent='Нет связи с сервером';});

