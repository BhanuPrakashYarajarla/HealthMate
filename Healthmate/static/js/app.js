// app.js
document.addEventListener('DOMContentLoaded', () => {
    const sendButton = document.getElementById('sendButton');
    const messageInput = document.getElementById('message');
    const responseDiv = document.getElementById('response');

    sendButton.addEventListener('click', sendMessage);

    function sendMessage() {
        const message = messageInput.value.trim();
        if (!message) {
            responseDiv.innerHTML = '<p>Please enter a message.</p>';
            return;
        }

        fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok ' + response.statusText);
            }
            return response.json();
        })
        .then(data => {
            if (data.table) {
                const table = createTable(data.table.headers, data.table.rows);
                responseDiv.innerHTML = table;
            } else {
                responseDiv.innerHTML = '<p>No response received.</p>';
            }
        })
        .catch(error => {
            console.error('Error:', error);
            responseDiv.innerHTML = '<p>An error occurred. Check the console for details.</p>';
        });
    }

    function createTable(headers, rows) {
        let tableHTML = '<table border="1" style="border-collapse: collapse; width: 50%;margin-top:35px;margin-left:25%;">';
        tableHTML += '<thead><tr>';
        headers.forEach(header => {
            tableHTML += `<th style="padding: 8px; text-align: left;">${header}</th>`;
        });
        tableHTML += '</tr></thead><tbody>';

        rows.forEach(row => {
            tableHTML += '<tr>';
            row.forEach(cell => {
                tableHTML += `<td style="padding: 8px;">${cell}</td>`;
            });
            tableHTML += '</tr>';
        });

        tableHTML += '</tbody></table>';
        return tableHTML;
    }
});
