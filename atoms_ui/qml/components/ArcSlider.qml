import QtQuick 2.15
import QtQuick.Controls 2.15
import ".." as Theme

Item {
    id: root
    property real from: 0
    property real to: 100
    property real value: 0
    property color color: Theme.Theme.teal
    property string label: ""
    signal valueChanged(real v)

    width: 100; height: 60

    Canvas {
        id: canvas
        anchors.fill: parent
        onPaint: {
            var ctx = getContext("2d");
            ctx.clearRect(0,0,width,height);
            ctx.lineWidth = 6;
            ctx.strokeStyle = Theme.Theme.border;
            ctx.beginPath();
            ctx.arc(width/2, height, width/2-6, Math.PI, 2*Math.PI);
            ctx.stroke();
            ctx.strokeStyle = root.color;
            var ang = Math.PI + (root.value-root.from)/(root.to-root.from)*Math.PI;
            ctx.beginPath();
            ctx.arc(width/2, height, width/2-6, Math.PI, ang);
            ctx.stroke();
        }
    }

    Text {
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.top: parent.bottom
        anchors.topMargin: 4
        text: root.label
        color: Theme.Theme.text
        font.pixelSize: 12
    }

    MouseArea {
        anchors.fill: parent
        onPositionChanged: update(mouse.x, mouse.y)
        onPressed: update(mouse.x, mouse.y)
        function update(mx, my) {
            var dx = mx - width/2;
            var dy = height - my;
            var ang = Math.atan2(dy, dx);
            if (ang < 0) ang = 0;
            if (ang > Math.PI) ang = Math.PI;
            root.value = root.from + (ang/Math.PI)*(root.to-root.from);
            canvas.requestPaint();
            root.valueChanged(root.value);
        }
    }
    onValueChanged: canvas.requestPaint()
    onColorChanged: canvas.requestPaint()
    Component.onCompleted: canvas.requestPaint()
}
