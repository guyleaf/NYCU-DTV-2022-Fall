const entry = () => {
  const video = document.getElementById("video");
  const videoLoadingAnimation = document.getElementById("video-loading-animation");
  const canvas = document.getElementById("canvas");
  const fps = document.getElementById("fps");
  const playBtn = document.getElementById("play-button");
  const pauseBtn = document.getElementById("pause-button");
  const reloadBtn = document.getElementById("reload-button");

  let streamer = null;
  let states = {
    droppedFrames: 0,
    fps: 0,
    totalFrames: 0
  };

  if (Hls.isSupported()) {
    const options = {
      src: video.src,
      videoElement: video
    };
    streamer = new HlsTech(options);
    streamer.start();
  } else if (video.canPlayType("application/vnd.apple.mpegurl")) {
    video.play();
  }

  setInterval(() => {
    let videoPlaybackQuality = video.getVideoPlaybackQuality();
    states.droppedFrames = videoPlaybackQuality.droppedVideoFrames;
    states.fps = (videoPlaybackQuality.totalVideoFrames - states.totalFrames);
    states.totalFrames = videoPlaybackQuality.totalVideoFrames;
    fps.textContent = states.fps.toString().padStart(2, "0");
  }, 1000);

  video.addEventListener("playing", () => {
    videoLoadingAnimation.hidden = true;
    playBtn.disabled = true;
    playBtn.classList.add("is-disabled");
    pauseBtn.disabled = false;
    pauseBtn.classList.remove("is-disabled");
  });

  video.addEventListener("pause", () => {
    playBtn.disabled = false;
    playBtn.classList.remove("is-disabled");
    pauseBtn.disabled = true;
    pauseBtn.classList.add("is-disabled");
  })

  playBtn.addEventListener("click", () => {
    streamer?.startLoad();
    video.play();
  });

  pauseBtn.addEventListener("click", () => {
    video.pause();
    streamer?.stopLoad();
  });

  reloadBtn.addEventListener("click", () => {
    videoLoadingAnimation.hidden = false;
    streamer?.reload();
    video.play()
    .then(() => {
      videoLoadingAnimation.hidden = true;
    });
  });
};

document.addEventListener("DOMContentLoaded", entry);