import QtQuick 2.15
import QtQuick.Controls 2.15
import ".." as Theme

CheckBox {
    id: root
    property alias label: textLabel.text
    indicator: Rectangle {
        implicitWidth: 40
        implicitHeight: 20
        radius: height/2
        border.color: Theme.Theme.border
        color: root.checked ? Theme.Theme.teal : Theme.Theme.panel
        Rectangle {
            width: 18; height: 18
            radius: 9
            color: Theme.Theme.bg
            anchors.verticalCenter: parent.verticalCenter
            x: root.checked ? parent.width-20 : 2
        }
    }
    contentItem: Text {
        id: textLabel
        text: ""
        color: Theme.Theme.text
        anchors.centerIn: parent
        font.pixelSize: 10
    }
}
