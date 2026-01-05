// View Structure function
function viewStructure(jsonFilename) {
    // Open viewer in new window
    const url = `/view/current/${jsonFilename}`;
    window.open(url, '_blank', 'width=1200,height=800');
}
