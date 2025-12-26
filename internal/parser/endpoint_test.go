package parser

import (
	"testing"
)

func TestSplitIntoBatches(t *testing.T) {
	tests := []struct {
		name      string
		files     []fileWithContent
		maxBytes  int
		wantCount int // バッチ数
	}{
		{
			name:      "empty files",
			files:     []fileWithContent{},
			maxBytes:  1000,
			wantCount: 0,
		},
		{
			name: "single small file",
			files: []fileWithContent{
				{path: "a.ts", content: "small", size: 5},
			},
			maxBytes:  1000,
			wantCount: 1,
		},
		{
			name: "multiple files fit in one batch",
			files: []fileWithContent{
				{path: "a.ts", content: "aaa", size: 100},
				{path: "b.ts", content: "bbb", size: 100},
				{path: "c.ts", content: "ccc", size: 100},
			},
			maxBytes:  500,
			wantCount: 1,
		},
		{
			name: "files split into multiple batches",
			files: []fileWithContent{
				{path: "a.ts", content: "aaa", size: 100},
				{path: "b.ts", content: "bbb", size: 100},
				{path: "c.ts", content: "ccc", size: 100},
				{path: "d.ts", content: "ddd", size: 100},
			},
			maxBytes:  250,
			wantCount: 2,
		},
		{
			name: "single large file exceeds max",
			files: []fileWithContent{
				{path: "large.ts", content: "large content", size: 2000},
			},
			maxBytes:  1000,
			wantCount: 1, // 大きいファイルも1バッチとして処理
		},
		{
			name: "mixed sizes with large file",
			files: []fileWithContent{
				{path: "a.ts", content: "aaa", size: 100},
				{path: "large.ts", content: "large", size: 2000},
				{path: "b.ts", content: "bbb", size: 100},
			},
			maxBytes:  500,
			wantCount: 3, // small, large, small
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			batches := splitIntoBatches(tt.files, tt.maxBytes)
			if len(batches) != tt.wantCount {
				t.Errorf("splitIntoBatches() got %d batches, want %d", len(batches), tt.wantCount)
			}

			// 全ファイルがバッチに含まれているか確認
			var totalFiles int
			for _, batch := range batches {
				totalFiles += len(batch)
			}
			if totalFiles != len(tt.files) {
				t.Errorf("total files in batches = %d, want %d", totalFiles, len(tt.files))
			}
		})
	}
}

func TestSplitIntoBatches_BatchSizeLimit(t *testing.T) {
	// 各バッチがmaxBytesを超えないことを確認
	files := []fileWithContent{
		{path: "a.ts", size: 100},
		{path: "b.ts", size: 150},
		{path: "c.ts", size: 200},
		{path: "d.ts", size: 100},
		{path: "e.ts", size: 150},
	}
	maxBytes := 300

	batches := splitIntoBatches(files, maxBytes)

	for i, batch := range batches {
		var batchSize int
		for _, fc := range batch {
			batchSize += fc.size
		}
		// 単一ファイルが大きい場合を除き、バッチサイズは制限以下
		if len(batch) > 1 && batchSize > maxBytes {
			t.Errorf("batch %d size = %d, exceeds max %d", i, batchSize, maxBytes)
		}
	}
}
