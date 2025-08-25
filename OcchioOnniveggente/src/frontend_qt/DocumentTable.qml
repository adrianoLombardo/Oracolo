import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

ListView {
    id: table
    property var documents: []
    property string filterText: ""
    property var selectedDocument: null

    model: ListModel { id: docModel }

    function refreshModel() {
        docModel.clear()
        for (var i = 0; i < documents.length; ++i) {
            var d = documents[i]
            if (filterText === "" ||
                (d.title && d.title.toLowerCase().indexOf(filterText.toLowerCase()) !== -1)) {
                docModel.append(d)
            }
        }
        if (currentIndex >= docModel.count) {
            selectedDocument = null
        }
    }

    onDocumentsChanged: refreshModel()
    onFilterTextChanged: refreshModel()

    delegate: Rectangle {
        width: table.width
        height: 28
        color: ListView.isCurrentItem ? "#1e1e1e" : "transparent"
        RowLayout {
            anchors.fill: parent
            spacing: 8
            Text { text: model.title; Layout.preferredWidth: 200 }
            Text { text: model.domain; Layout.preferredWidth: 120 }
            Text { text: model.rule; Layout.preferredWidth: 120 }
            Text { text: model.status; Layout.preferredWidth: 100 }
            Text { text: model.confidence; Layout.preferredWidth: 60 }
        }
        MouseArea {
            anchors.fill: parent
            onClicked: {
                table.currentIndex = index
                table.selectedDocument = model
            }
        }
    }

    ScrollBar.vertical: ScrollBar {}
}
