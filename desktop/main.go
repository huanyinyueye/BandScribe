// BandScribe's development desktop host. The existing Python service owns the
// transcription pipeline and project data; MyGo owns the desktop window.
package main

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"time"

	"github.com/egoist/mygo"
)

// backend finds the Python service: the packaged layout (python\ and app\ next to the
// executable, see scripts/package_app.ps1) or the development checkout one level up.
func backend() (python, root string, packaged bool) {
	if exe, err := os.Executable(); err == nil {
		dir := filepath.Dir(exe)
		python = filepath.Join(dir, "python", "pythonw.exe")
		if _, err := os.Stat(python); err == nil {
			return python, filepath.Join(dir, "app"), true
		}
	}
	root, err := filepath.Abs("..")
	if err != nil {
		log.Fatal(err)
	}
	python = filepath.Join(root, ".venv", "Scripts", "python.exe")
	if _, err := os.Stat(python); err != nil {
		log.Fatalf("BandScribe environment missing: %v; run py -m uv sync first", err)
	}
	return python, root, false
}

func main() {
	python, root, packaged := backend()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		log.Fatal(err)
	}
	port := listener.Addr().(*net.TCPAddr).Port
	listener.Close()
	url := fmt.Sprintf("http://127.0.0.1:%d", port)
	secret := make([]byte, 32)
	if _, err := rand.Read(secret); err != nil {
		log.Fatal(err)
	}
	token := hex.EncodeToString(secret)
	server := exec.Command(python, "-m", "bandscribe.cli", "gui", "--no-window", "--port", fmt.Sprint(port))
	server.Dir = root
	server.Env = append(os.Environ(), "BANDSCRIBE_SHUTDOWN_TOKEN="+token)
	server.Stdout, server.Stderr = os.Stdout, os.Stderr
	if cache, err := os.UserCacheDir(); packaged && err == nil {
		// The packaged host has no console: keep the backend's output for diagnosis.
		if err := os.MkdirAll(filepath.Join(cache, "BandScribe"), 0o755); err == nil {
			if f, err := os.Create(filepath.Join(cache, "BandScribe", "backend.log")); err == nil {
				defer f.Close()
				server.Stdout, server.Stderr = f, f
			}
		}
	}
	if err := server.Start(); err != nil {
		log.Fatal(err)
	}
	defer func() {
		client := &http.Client{Timeout: 5 * time.Second}
		request, err := http.NewRequest(http.MethodPost, url+"/api/shutdown", nil)
		if err != nil {
			log.Print(err)
			return
		}
		request.Header.Set("X-BandScribe-Shutdown", token)
		response, err := client.Do(request)
		if err == nil {
			response.Body.Close()
		}
		// A running transcription must finish its current step and write its
		// cancellation manifest before its Python process is reaped.
		if err == nil && response.StatusCode == http.StatusOK {
			_ = server.Wait()
		} else {
			log.Print("Could not request a clean backend shutdown; leaving it running to protect active work")
		}
	}()

	client := &http.Client{Timeout: time.Second}
	ready := false
	// A cold start imports numpy, scipy, librosa and ONNX Runtime; allow for slow disks.
	for deadline := time.Now().Add(120 * time.Second); time.Now().Before(deadline); {
		response, err := client.Get(url + "/api/info")
		if err == nil {
			response.Body.Close()
			if response.StatusCode == http.StatusOK {
				ready = true
				break
			}
		}
		time.Sleep(200 * time.Millisecond)
	}
	if !ready {
		log.Print("BandScribe backend did not start")
		return
	}
	mygo.App.WhenReady(func() {
		mygo.NewWindow(mygo.WindowOptions{Title: "BandScribe", Width: 1400, Height: 900, URL: url})
	})
	if err := mygo.App.Run(); err != nil {
		log.Print(err)
	}
}
