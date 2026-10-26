
"use strict";
const data=JSON.parse(document.getElementById("paper-data").textContent);
const $=id=>document.getElementById(id);
const tabs=[...document.querySelectorAll("[data-tab]")];
function showTab(id){
 tabs.forEach(button=>{const on=button.dataset.tab===id;button.setAttribute("aria-selected",String(on));button.tabIndex=on?0:-1;});
 document.querySelectorAll(".panel").forEach(panel=>panel.hidden=panel.id!==id);
 if(id==="geometry")renderGeometry();
}
tabs.forEach((button,index)=>{
 button.addEventListener("click",()=>showTab(button.dataset.tab));
 button.addEventListener("keydown",event=>{
  let next=index;
  if(event.key==="ArrowRight")next=(index+1)%tabs.length;
  else if(event.key==="ArrowLeft")next=(index+tabs.length-1)%tabs.length;
  else if(event.key==="Home")next=0;
  else if(event.key==="End")next=tabs.length-1;
  else return;
  event.preventDefault();showTab(tabs[next].dataset.tab);tabs[next].focus();
 });
});
function setMode(training){
 $("training").setAttribute("aria-pressed",String(training));
 $("deployment").setAttribute("aria-pressed",String(!training));
 $("auxiliary").hidden=!training;
 $("mode-label").textContent=training?"训练阶段":"部署阶段";
 $("mode-title").textContent=training?"标签给梯度，部署输入不增加":"删除 critic 和 decoder，actor 继续工作";
 $("mode-description").textContent=training?"decoder 从 latent 预测接触类别，损失更新编码器；接触真值和预测都不作为 actor 的输入。":"推理只使用可部署的本体感知历史。接触监督已经改变编码器参数，不需要实时提供接触标签或 decoder 预测。";
}
$("training").addEventListener("click",()=>setMode(true));
$("deployment").addEventListener("click",()=>setMode(false));
let task="peg",condition=6;
const numeric=value=>Number(String(value).replace(">",""));
const formatted=value=>typeof value==="string"?value:Number(value).toFixed(1);
function renderResults(){
 const item=data.tasks[task];
 document.querySelectorAll(".task").forEach(button=>button.setAttribute("aria-pressed",String(button.dataset.task===task)));
 $("generalization-chart").replaceChildren();
 data.categories.forEach((label,index)=>{
  const row=document.createElement("button");row.className="bar-group";row.type="button";
  row.setAttribute("aria-pressed",String(index===condition));
  const heading=document.createElement("div");heading.className="bar-heading";
  const title=document.createElement("span");title.textContent=label;
  const hint=document.createElement("small");hint.textContent=index===0?"训练范围":"单属性变化";
  heading.append(title,hint);row.append(heading);
  [["blind","本体感知",""],["visual","仅RGB","orange"]].forEach(([key,name,color])=>{
   const line=document.createElement("div");line.className="bar-line";
   const track=document.createElement("div");track.className="track";
   const fill=document.createElement("div");fill.className="fill "+color;fill.style.width=numeric(item[key][index])+"%";
   track.append(fill);
   const value=document.createElement("span");value.className="bar-value";value.textContent=formatted(item[key][index])+"%";
   line.append(track,value);row.append(line);
  });
  row.setAttribute("aria-label",label+"：本体感知 "+formatted(item.blind[index])+"%，RGB "+formatted(item.visual[index])+"%");
  row.addEventListener("click",()=>{condition=index;renderResults();});
  $("generalization-chart").append(row);
 });
 const b=item.blind[condition],v=item.visual[condition],gap=numeric(b)-v;
 $("condition-name").textContent=item.name+" · "+data.categories[condition];
 $("condition-title").textContent="成功率 "+formatted(b)+"% / "+formatted(v)+"%";
 $("condition-gap").textContent=(typeof b==="string"?"> ":"")+(gap>=0?"+":"")+gap.toFixed(1)+" pp";
 $("condition-description").textContent=item.conditions[condition];
 $("condition-interpretation").textContent=task==="peg"&&condition===6?"优势为 32.3 个百分点，但盲策略仍失败超过一半。相对优势和绝对可靠性要分别判断。":task==="nut"&&condition===3?"这里视觉略高。表7并不支持“每个几何变化都由盲策略胜出”。":typeof b==="string"?"作者报告 >99.9%，图形用 99.9% 下界绘制。差值同样是下界，不当作精确 100%。":"pp 表示百分点差，由表7报告值相减，不代表相对百分比提升。";
 $("seed-note").textContent="来源：表6–7，p.20。"+(item.seeds===2?"插销为两组训练 seed 的平均。":"本任务仅一组训练 seed。")+"双方均有30步历史；图中没有推断未报告的置信区间。";
 $("speed-description").textContent=item.speed;
}
document.querySelectorAll(".task").forEach(button=>button.addEventListener("click",()=>{task=button.dataset.task;renderResults();}));
function wilson(k,n){
 const z=1.959963984540054,p=k/n,den=1+z*z/n;
 const center=(p+z*z/(2*n))/den,half=z*Math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den;
 return [100*(center-half),100*(center+half)];
}
function renderReal(){
 const choice=$("real-select").value;
 const selected=data.real.filter(item=>choice==="all"||item.group===choice||item.id===choice);
 const k=selected.reduce((a,item)=>a+item.success,0),n=selected.reduce((a,item)=>a+item.trials,0),ci=wilson(k,n);
 $("real-chart").replaceChildren();
 data.real.forEach(item=>{
  const row=document.createElement("div");row.className="real-row"+(selected.includes(item)?" selected":"");
  const label=document.createElement("span");label.textContent=item.id;
  const track=document.createElement("div");track.className="track";
  const fill=document.createElement("div");fill.className="fill";fill.style.width=100*item.success/item.trials+"%";track.append(fill);
  const value=document.createElement("span");value.className="count";value.textContent=item.success+" / "+item.trials;
  row.append(label,track,value);$("real-chart").append(row);
 });
 $("real-label").textContent=$("real-select").selectedOptions[0].textContent;
 $("real-rate").textContent=(100*k/n).toFixed(1)+"%";
 $("real-count").textContent=k+" / "+n+" 次成功";
 $("real-ci").textContent="95% Wilson 区间："+ci[0].toFixed(1)+"%–"+ci[1].toFixed(1)+"%。由报告计数计算。合并区间未建模零件与试验间的相关性。";
 $("real-condition").textContent=selected.length===1?selected[0].condition+"；单边间隙 "+selected[0].clearance.toFixed(2)+" mm。":"所选组含 "+selected.length+" 个零件；单边间隙 "+Math.min(...selected.map(x=>x.clearance)).toFixed(2)+"–"+Math.max(...selected.map(x=>x.clearance)).toFixed(2)+" mm。";
}
$("real-select").addEventListener("change",renderReal);
function renderGeometry(){
 const n=Number($("sides").value),c=Number($("clearance").value),radius=c*0.78,max=radius/Math.cos(Math.PI/n),e=c*(1/Math.cos(Math.PI/n)-1);
 $("sides-out").textContent=n;$("clearance-out").textContent=c+" μm";
 $("feasible").setAttribute("points",Array.from({length:n},(_,i)=>{const theta=(2*i+1)*Math.PI/n;return (280+max*Math.cos(theta)).toFixed(3)+","+(195-max*Math.sin(theta)).toFixed(3);}).join(" "));
 $("ideal").setAttribute("r",radius);$("radius-line").setAttribute("x2",280+radius);
 $("radius-label").textContent="c = "+c+" μm";
 $("extra-play").textContent=e.toFixed(4)+" μm";$("relative-play").textContent=(100*e/c).toFixed(4)+"%";
 $("geometry-svg").setAttribute("aria-label",n+"边外接多边形，径向间隙"+c+"微米，额外游隙"+e.toFixed(4)+"微米");
}
$("sides").addEventListener("input",renderGeometry);$("clearance").addEventListener("input",renderGeometry);
renderResults();renderReal();renderGeometry();
