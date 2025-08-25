import QtQuick
import "../Palette.js" as Palette

Item {
  id: orb
  property real radius: 100
  width: radius * 2
  height: width
  opacity: 0.2

  Rectangle {
    anchors.fill: parent
    radius: width / 2
    gradient: Gradient {
      GradientStop { position: 0.0; color: Palette.accent }
      GradientStop { position: 1.0; color: "transparent" }
    }

Item {
  id: root
  property real radius: 80
  width: radius*2
  height: radius*2

  ShaderEffect {
    anchors.fill: parent
    property real t: 0
    fragmentShader: """
      uniform lowp float qt_Opacity;
      uniform float t;
      varying highp vec2 qt_TexCoord0;
      void main() {
          vec2 uv = qt_TexCoord0 * 2.0 - 1.0;
          float r = length(uv);
          float angle = atan(uv.y, uv.x);
          float glow = smoothstep(1.0, 0.8, r);
          vec3 color1 = vec3(0.0, 0.9, 1.0);
          vec3 color2 = vec3(0.4, 1.0, 0.9);
          float mixv = 0.5 + 0.5 * sin(angle * 3.0 + t);
          vec3 color = mix(color1, color2, mixv);
          gl_FragColor = vec4(color, glow) * qt_Opacity;
      }
    """
    NumberAnimation on t { from: 0; to: 6.28318; duration: 4000; loops: Animation.Infinite }
  }
}
