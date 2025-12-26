package parser

import (
	"os"
	"path/filepath"
	"testing"
)

func TestResolveRelatedPath(t *testing.T) {
	tests := []struct {
		name     string
		codeDir  string
		relFile  string
		expected string
	}{
		{
			name:     "relFile が codeDir で始まる場合は結合しない",
			codeDir:  "src",
			relFile:  "src/client/components/Page.tsx",
			expected: "src/client/components/Page.tsx",
		},
		{
			name:     "relFile が codeDir で始まらない場合は結合する",
			codeDir:  "src",
			relFile:  "client/components/Page.tsx",
			expected: "src/client/components/Page.tsx",
		},
		{
			name:     "codeDir がスラッシュ付きの場合",
			codeDir:  "src/",
			relFile:  "src/client/components/Page.tsx",
			expected: "src/client/components/Page.tsx",
		},
		{
			name:     "通常の結合",
			codeDir:  "src",
			relFile:  "utils/helper.ts",
			expected: "src/utils/helper.ts",
		},
		{
			name:     "空の codeDir",
			codeDir:  "",
			relFile:  "client/components/Page.tsx",
			expected: "client/components/Page.tsx",
		},
		{
			name:     "relFile が空の場合",
			codeDir:  "src",
			relFile:  "",
			expected: "src",
		},
		{
			name:     "両方空の場合",
			codeDir:  "",
			relFile:  "",
			expected: ".",
		},
		{
			name:     "ドットを含むパス",
			codeDir:  "src",
			relFile:  "./client/components/Page.tsx",
			expected: "src/client/components/Page.tsx",
		},
		{
			name:     "relFile に余分なスラッシュがある場合",
			codeDir:  "src",
			relFile:  "src//client/components/Page.tsx",
			expected: "src/client/components/Page.tsx",
		},
		{
			name:     "codeDir と同じ名前のファイル",
			codeDir:  "src",
			relFile:  "src",
			expected: "src",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := resolveRelatedPath(tt.codeDir, tt.relFile)
			// filepath.Clean で正規化して比較
			expected := filepath.Clean(tt.expected)
			resultClean := filepath.Clean(result)
			if resultClean != expected {
				t.Errorf("resolveRelatedPath(%q, %q) = %q, want %q", tt.codeDir, tt.relFile, result, tt.expected)
			}
		})
	}
}

func TestFindCodeFilesWithCodePaths(t *testing.T) {
	// テスト用の一時ディレクトリを作成
	tmpDir, err := os.MkdirTemp("", "spec-verify-test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	// テスト用のディレクトリ構造を作成
	// src/client/components/pages/ImageSynthesisPage.tsx
	componentDir := filepath.Join(tmpDir, "src", "client", "components", "pages")
	if err := os.MkdirAll(componentDir, 0755); err != nil {
		t.Fatalf("Failed to create component dir: %v", err)
	}

	testFile := filepath.Join(componentDir, "ImageSynthesisPage.tsx")
	if err := os.WriteFile(testFile, []byte("// test component"), 0644); err != nil {
		t.Fatalf("Failed to create test file: %v", err)
	}

	codeDir := filepath.Join(tmpDir, "src")

	tests := []struct {
		name         string
		spec         *Spec
		codeDir      string
		codePaths    []string
		wantContains string
	}{
		{
			name: "spec_types.code_paths を使用して関連ファイルを検索",
			spec: &Spec{
				Type: "ui",
				// codeDir 相対のパスを指定（src/ は付けない）
				RelatedFiles: []string{"client/components/pages/ImageSynthesisPage"},
			},
			codeDir:      codeDir,
			codePaths:    []string{filepath.Join(codeDir, "client", "components")},
			wantContains: "ImageSynthesisPage.tsx",
		},
		{
			name: "絶対パス指定の場合",
			spec: &Spec{
				Type: "ui",
				// 絶対パスを直接指定
				RelatedFiles: []string{testFile},
			},
			codeDir:      codeDir,
			codePaths:    []string{filepath.Join(codeDir, "client", "components")},
			wantContains: "ImageSynthesisPage.tsx",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			files, err := FindCodeFilesWithCodePaths(tt.spec, tt.codeDir, tt.codePaths)
			if err != nil {
				t.Fatalf("FindCodeFilesWithCodePaths failed: %v", err)
			}

			// 期待するファイルが含まれているか確認
			found := false
			for _, f := range files {
				if filepath.Base(f) == tt.wantContains {
					found = true
					break
				}
			}

			if !found {
				t.Errorf("Expected file containing %q not found in results: %v", tt.wantContains, files)
			}
		})
	}
}

func TestFindCodeFiles_BackwardCompatibility(t *testing.T) {
	// テスト用の一時ディレクトリを作成
	tmpDir, err := os.MkdirTemp("", "spec-verify-test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	// テスト用のディレクトリ構造を作成
	componentDir := filepath.Join(tmpDir, "src", "client", "components")
	if err := os.MkdirAll(componentDir, 0755); err != nil {
		t.Fatalf("Failed to create component dir: %v", err)
	}

	testFile := filepath.Join(componentDir, "TestPage.tsx")
	if err := os.WriteFile(testFile, []byte("// test component"), 0644); err != nil {
		t.Fatalf("Failed to create test file: %v", err)
	}

	// 後方互換性テスト: mapping を使用
	spec := &Spec{
		Type:         "ui",
		RoutePath:    "/test",
		RelatedFiles: []string{},
	}

	mapping := map[string]string{
		"ui": "client/components",
	}

	files, err := FindCodeFiles(spec, filepath.Join(tmpDir, "src"), mapping)
	if err != nil {
		t.Fatalf("FindCodeFiles failed: %v", err)
	}

	// TestPage.tsx が見つかるはず（ルート名 "test" から推測）
	found := false
	for _, f := range files {
		if filepath.Base(f) == "TestPage.tsx" {
			found = true
			break
		}
	}

	if !found {
		t.Errorf("Expected TestPage.tsx not found in results: %v", files)
	}
}
