class HlsTech {
  constructor(_options) {
    let self = this;
    this._options = _options;
    this._video = _options.videoElement;

    this._config = {
      enableWorker: true,
      autoStartLoad: true,
      liveSyncDurationCount: 1,
      liveMaxLatencyDurationCount: 10,
      maxLiveSyncPlaybackRate: 1.8,
      liveDurationInfinity: true,
      lowLatencyMode: true,
    };

    this._player = new Hls(this._config);

    this._player.on(Hls.Events.MANIFEST_LOADED, function (event, data) {
      // self.startLoad();
      self._video.play();
    });

    this._player.on(Hls.Events.ERROR, function (eventName, data) {
      console.warn('Error event:', data);

      if (data.fatal) {
        switch (data.type) {
          case Hls.ErrorTypes.NETWORK_ERROR:
            // try to recover network error
            console.log('fatal network error encountered, try to recover');
            self.startLoad();
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
    this._player.loadSource(this._options.src);
    this._player.attachMedia(this._video);
  }

  startLoad() {
    this._player.startLoad(-1);
  }

  stopLoad() {
    this._player.stopLoad();
  }

  reload() {
    this.stopLoad();
    this.startLoad();
  }

  destroy() {
    if (this._player != null) {
      this._player.destroy();
      this._player = null;
    }
  };
}
