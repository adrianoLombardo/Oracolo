import QtQuick 2.15
import QtQuick.Controls 2.15
import ".." as Theme

Item {
    id: root
    width: 300
    height: 300

    Canvas {
        id: canvas
        anchors.fill: parent
        onPaint: {
            var ctx = getContext("2d");
            ctx.clearRect(0,0,width,height);
            ctx.strokeStyle = Theme.Theme.border;
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.arc(width/2, height/2, width/2-10, 0, 2*Math.PI);
            ctx.stroke();
        }
    }

    Repeater {
        model: 6
        Rectangle {
            width: 6; height: 6; radius: 3
            color: Theme.Theme.teal
            property real angle: index/6*360
            NumberAnimation on angle { from: angle; to: angle+360; duration: 8000; loops: Animation.Infinite }
            function toRad(a){ return a*Math.PI/180 }
            x: root.width/2 + Math.cos(toRad(angle))*(root.width/2-20) - width/2
            y: root.height/2 + Math.sin(toRad(angle))*(root.height/2-20) - height/2
        }
    }
}
