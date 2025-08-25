import QtQuick
import "../Palette.js" as Palette

Canvas {
  id: canvas
  property color colorLine: Palette.accent
  property var history: []
  signal tick()
  width: 240; height: 60

  function appendLevel(v) {
    if (history.length > 120) history.shift()
    history.push(v)
    requestAnimationFrame(paint)
  }

  function start() {
    // timer “soft” per demo (in un caso reale guida l’update da audio chunk)
    timer.start()
  }

  Timer {
    id: timer; interval: 50; repeat: true
    onTriggered: canvas.tick()
  }

  onPaint: {
    var ctx = getContext("2d")
    ctx.reset()
    ctx.clearRect(0,0,width,height)
    ctx.lineWidth = 2
    ctx.strokeStyle = colorLine
    ctx.beginPath()
    var n = history.length
    for (var i=0; i<n; ++i) {
      var x = i * (width / Math.max(1, n-1))
      var y = height - (history[i] * height)
      if (i === 0) ctx.moveTo(x,y); else ctx.lineTo(x,y)
    }
    ctx.stroke()
  }
}
