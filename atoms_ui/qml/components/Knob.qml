import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import ".." as Theme

Item {
    id: root
    property alias value: dial.value
    property real from: 0
    property real to: 100
    property string label: ""
    property bool motion: false

    signal valueChanged(real v)

    width: 100; height: 120

    Column {
        anchors.centerIn: parent
        spacing: 4
        Dial {
            id: dial
            from: root.from
            to: root.to
            value: (from + to)/2
            onValueChanged: { root.valueChanged(value); bg.requestPaint(); }
            background: Canvas {
                id: bg
                anchors.fill: parent
                onPaint: {
                    var ctx = getContext("2d");
                    ctx.clearRect(0, 0, width, height);
                    ctx.strokeStyle = Theme.Theme.teal;
                    ctx.lineWidth = 6;
                    var ang = (dial.value - dial.from)/(dial.to-dial.from)*Math.PI*1.5 + Math.PI*0.75;
                    ctx.beginPath();
                    ctx.arc(width/2, height/2, width/2-6, Math.PI*0.75, ang);
                    ctx.stroke();
                }
            }
        }
        Text {
            text: root.label
            anchors.horizontalCenter: parent.horizontalCenter
            color: Theme.Theme.text
            font.pixelSize: 12
        }
        TogglePill {
            id: toggle
            width: 50; height: 20
            checked: root.motion
            onCheckedChanged: root.motion = checked
            label: "MOTION"
        }
    }
    Component.onCompleted: bg.requestPaint()
}
