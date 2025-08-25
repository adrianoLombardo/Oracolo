import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
RowLayout {
    id: rules
    spacing: 12
    signal applyRules(var rules)

    CheckBox { id: verifiedBox; text: "Solo verificati" }
    CheckBox { id: citationsBox; text: "Citazioni" }
    CheckBox { id: piiBox; text: "Blocca PII" }
    Label { text: "Confidenza minima" }
    Slider { id: confidenceSlider; from: 0; to: 1; stepSize: 0.1; value: 0.5; Layout.preferredWidth: 120 }
    Button {
        text: "Applica"
        onClicked: rules.applyRules({
            use_verified: verifiedBox.checked,
            allow_citations: citationsBox.checked,
            block_pii: piiBox.checked,
            min_confidence: confidenceSlider.value
        })
    }
}
