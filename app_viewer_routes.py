@app.route('/view/<task_id>/<filename>')
def view_structure(task_id, filename):
    """Render structure viewer for a JSON file"""
    return render_template('viewer.html', task_id=task_id, filename=filename)


@app.route('/api/structure/<filename>')
def get_structure(filename):
    """Get JSON structure data"""
    try:
        filepath = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 404
