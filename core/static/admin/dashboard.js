
(function(){
  const THEME_KEY = "taw_dash_theme";
  const link = document.getElementById("bsTheme");
  const btn = document.getElementById("themeToggle");

  // Two premium themes: dark + light
  const THEMES = {
    dark: "https://cdn.jsdelivr.net/npm/bootswatch@5.3.3/dist/darkly/bootstrap.min.css",
    light: "https://cdn.jsdelivr.net/npm/bootswatch@5.3.3/dist/lux/bootstrap.min.css"
  };

  function apply(theme){
    const t = theme === "light" ? "light" : "dark";
    if(link) link.href = THEMES[t];
    document.body.dataset.theme = t;
    try{ localStorage.setItem(THEME_KEY, t); }catch(e){}
    if(btn){
      btn.innerHTML = t === "light"
        ? '<i class="bi bi-moon-stars"></i><span class="ms-1">Dark</span>'
        : '<i class="bi bi-sun"></i><span class="ms-1">Light</span>';
    }
  }

  function init(){
    let t = "dark";
    try{
      const saved = localStorage.getItem(THEME_KEY);
      if(saved) t = saved;
    }catch(e){}
    apply(t);

    if(btn){
      btn.addEventListener("click", function(){
        const cur = document.body.dataset.theme || "dark";
        apply(cur === "dark" ? "light" : "dark");
      });
    }
  }

  if(document.readyState === "loading"){
    document.addEventListener("DOMContentLoaded", init);
  }else{
    init();
  }
})();
