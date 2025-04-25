document.addEventListener("DOMContentLoaded", async () => {
  const container = document.getElementById("policies");
  const popup = document.getElementById("video-popup");
  const video = document.getElementById("popup-video");
  const prompt = document.getElementById("popup-prompt");

  const res = await fetch("/policy_analysis.json");
  const data = await res.json();

  data.forEach(policy => {
    const div = document.createElement("div");
    div.className = "policy";

    const header = document.createElement("div");
    header.className = "policy-header";
    header.innerHTML = `<strong class="policy-name">${policy.policy_name}</strong><button class="toggle-button">+</button>`;

    const summary = document.createElement("pre");
    summary.className = "policy-summary";
    summary.textContent = policy.summary;

    const full = document.createElement("div");
    full.className = "policy-full hidden";
    const fullTitle = document.createElement("div");
    fullTitle.textContent = "Full Report";
    fullTitle.className = "full-title";

    const fullContent = document.createElement("div");
    fullContent.innerHTML = policy.full_report.replace(/<ref>(.*?)<\/ref>/g, (match, sid) => {
      const videoPath = policy.session_id_to_video_path[sid];
      const taskPrompt = policy.session_id_to_prompt[sid];
      const shortSid = sid.split("-")[0];
      if (!videoPath) {
        return `<span class='hover-ref' data-video='' data-prompt='${taskPrompt || ""}'>🎥 ${shortSid}</span>`;
      }
      return `<span class='hover-ref' data-video='${videoPath}' data-prompt='${taskPrompt}'>🎥 ${shortSid}</span>`;
    });

    full.appendChild(fullTitle);
    full.appendChild(fullContent);

    header.addEventListener("click", () => {
      const isHidden = full.classList.contains("hidden");
      full.classList.toggle("hidden", !isHidden);
      header.querySelector("button").textContent = isHidden ? "−" : "+";
    });

    div.appendChild(header);
    summary.classList.add("summary-spaced");
    div.appendChild(summary);
    div.appendChild(full);
    container.appendChild(div);
  });

  document.body.addEventListener("mouseover", e => {
    if (e.target.classList.contains("hover-ref")) {
      const videoPath = e.target.dataset.video;
      const taskPrompt = e.target.dataset.prompt;

      if (videoPath) {
        video.src = `/videos/${videoPath}`;
        prompt.textContent = `Prompt: ${taskPrompt}`;
        popup.classList.remove("hidden");
        video.play();
      }
    }
  });

  document.body.addEventListener("mouseout", e => {
    if (e.target.classList.contains("hover-ref")) {
      popup.classList.add("hidden");
      video.pause();
      video.src = "";
    }
  });
});
