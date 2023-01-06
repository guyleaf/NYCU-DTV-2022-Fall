class HlsTech {
  constructor(_options) {
    let self = this;
    this._isLive = false;
    this._options = _options;
    this._states = {
      droppedFrames: 0,
      fps: 0,
      totalFrames: 0
    }

    this._config = {
      enableWorker: true,
      liveSyncDurationCount: 1,
      liveMaxLatencyDurationCount: 30,
      maxLiveSyncPlaybackRate: 2,
      liveDurationInfinity: true,
      lowLatencyMode: true,
    };

    this._player = new Hls(this._config);

    this._player.on(Hls.Events.LEVEL_LOADED, function (event, data) {
      if (data.details != undefined && data.details.type !== 'VOD') {
        self._isLive = true;
      }
    });

    this._player.on(Hls.Events.MANIFEST_LOADED, function (event, data) {
      self._options.videoElement.play();
    });

    this._player.on(Hls.Events.ERROR, function (eventName, data) {
      console.warn('Error event:', data);

      if (data.fatal) {
        switch (data.type) {
          case Hls.ErrorTypes.NETWORK_ERROR:
            // try to recover network error
            console.log('fatal network error encountered, try to recover');
            self._player.startLoad();
            break;
          case Hls.ErrorTypes.MEDIA_ERROR:
            console.log('fatal media error encountered, try to recover');
            self._player.recoverMediaError();
            self._player.swapAudioCodec();
            break;
          default:
            console.error("Unrecoverable error");
            self.destroy();
            break;
        }
      }
    });
  }

  get options() {
    return this._options;
  };

  get player() {
    return this._player;
  };

  get isLive() {
    return this._isLive;
  };

  getQualities() {
    var u = this._player.levels;
    var bitrates = [];

    for (var i = 0; i < u.length; i++) {
      var b = {};
      b.index = u[i].level != undefined ? u[i].level : i;
      b.bitrate = u[i].bitrate;
      b.height = u[i].height;
      b.width = u[i].width;
      b.resolution = u[i].attrs.RESOLUTION
      b.bane = u[i].name;
      bitrates.push(b);
    }

    return bitrates;
  };

  setQuality(index) {
    index = parseInt(index);
    this._player.currentLevel = index;
  };

  setMaxQuality() {
    var qualities = this.getQualities();
    maxQualityIndex = -1;
    bitrate = 0;

    for (var i = 0; i < qualities.length; i++) {
      if (qualities[i].bitrate > bitrate) {
        bitrate = qualities[i].bitrate;
        maxQualityIndex = i;
      }
    }

    this.setQuality(maxQualityIndex);
  };

  start() {
    let self = this;
    this._player.loadSource(this._options.src);
    this._player.attachMedia(this._options.videoElement);
    setInterval(() => {
      let videoPlaybackQuality = self._options.videoElement.getVideoPlaybackQuality();
      self._states.droppedFrames = videoPlaybackQuality.droppedVideoFrames;
      self._states.fps = (videoPlaybackQuality.totalVideoFrames - self._states.totalFrames);
      self._states.totalFrames = videoPlaybackQuality.totalVideoFrames;
      self._options.fpsElement.textContent = self._states.fps;
    }, 1000);
  }

  destroy() {
    if (this._player != null) {
      this._player.destroy();
      this._player = null;
    }
  };
}
